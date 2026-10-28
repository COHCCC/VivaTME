"""Pooled DiSCoTME training on all sections of a cohort (symmetric multi-positive InfoNCE, count-matched mixed negatives).

Launch with torchrun. All-gather is differentiable; local-anchor mean losses
are NOT multiplied by world size: DDP's parameter-gradient averaging supplies
the global-anchor mean. Repeated sampled spots form multiple positive pairs.
"""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import signal as signal_module
import sys
import time
import traceback
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, Sampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from encoder.dataset import CohortDataset, RGBTrainingDataset, collate_training
from encoder.loss import distributed_multi_positive_loss


def rank_world():
    return (dist.get_rank(), dist.get_world_size()) if dist.is_initialized() else (0, 1)


class DistributedBalancedSampler(Sampler):
    """Deterministic global weighted draws, then equal-length rank shards.

    Every global batch contains exactly world_size * batch_size anchors.
    Repeated global indices are legal and handled as multiple positives.
    """
    def __init__(self, dataset, batch_size, num_replicas=1, rank=0, seed=42,
                 steps_per_epoch=None):
        if batch_size < 1 or num_replicas < 1 or not 0 <= rank < num_replicas:
            raise ValueError('Invalid distributed batch parameters')
        self.batch_size, self.num_replicas, self.rank = int(batch_size), int(num_replicas), int(rank)
        self.seed, self.epoch = int(seed), 0
        self.start_step = 0
        global_batch = self.batch_size*self.num_replicas
        self.steps_per_epoch = math.ceil(len(dataset)/global_batch) if steps_per_epoch is None else int(steps_per_epoch)
        if self.steps_per_epoch < 1:
            raise ValueError('steps_per_epoch must be positive')
        self.global_count = self.steps_per_epoch*global_batch
        sizes = np.asarray(dataset.sample_sizes, dtype=np.float64)
        labels = np.asarray(dataset.sample_indices, dtype=np.int64)
        if not len(labels) or (sizes <= 0).any():
            raise ValueError('Balanced sampler needs nonempty samples')
        self.weights = torch.as_tensor(1.0/sizes[labels], dtype=torch.double)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return (self.steps_per_epoch-self.start_step)*self.batch_size

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed+self.epoch)
        indexes = torch.multinomial(self.weights, self.global_count, replacement=True, generator=generator)
        return iter(indexes[self.rank::self.num_replicas][self.start_step*self.batch_size:].tolist())


def build_model(model_config, gene_dim, device):
    """Common constructor for training, probe, and embedding extraction."""
    from encoder.model import PooledDiSCoTME
    kwargs = dict(model_config)
    if 'gene_dim' in kwargs and int(kwargs['gene_dim']) != gene_dim:
        raise ValueError('Configured gene_dim does not match input gene list')
    kwargs['gene_dim'] = int(gene_dim)
    return PooledDiSCoTME(**kwargs).to(device)


def optimizer_groups(model, training):
    rates = {'backbone':float(training.get('lr_backbone',1e-5)),
             'image':float(training.get('lr_image',1e-4)),
             'gene_encoder':float(training.get('lr_gene_encoder',3e-4)),
             'gene_projection':float(training.get('lr_gene_projection',1e-4))}
    groups = {key:[] for key in rates}
    names = {key:[] for key in rates}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith('backbone.'): key='backbone'
        elif name.startswith(('image_proj.','image_context.','img_head.')): key='image'
        elif name.startswith(('gene_encoder.','gene_context.')): key='gene_encoder'
        elif name.startswith('gene_head.'): key='gene_projection'
        else: raise ValueError(f'Unassigned trainable parameter: {name}')
        groups[key].append(parameter); names[key].append(name)
    if any(not parameters for parameters in groups.values()):
        raise ValueError('An expected optimizer group is empty')
    return [dict(params=groups[key],lr=rates[key],name=key) for key in rates], names


def model_state_sha256(state):
    h=hashlib.sha256()
    for name in sorted(state):
        value=state[name].detach().cpu().contiguous()
        h.update(name.encode());h.update(str(tuple(value.shape)).encode());h.update(str(value.dtype).encode())
        h.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def seed_worker(worker_id):
    worker_seed = torch.initial_seed()%2**32
    np.random.seed(worker_seed); random.seed(worker_seed)


def atomic_json(path, content):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(content,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    temp.replace(path)


def rng_state():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    if state['cuda'] is not None and torch.cuda.is_available(): torch.cuda.set_rng_state(state['cuda'])


def synchronized_bad(flag, signal):
    signal.fill_(int(flag))
    if dist.is_initialized(): dist.all_reduce(signal,op=dist.ReduceOp.MAX)
    return bool(signal.item())


def save_checkpoint(path, model, optimizer, scheduler, scaler, state, rank, world):
    local_rng=rng_state()
    rngs=[None]*world
    if world>1: dist.all_gather_object(rngs,local_rng)
    else: rngs=[local_rng]
    if rank==0:
        payload={**state,'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                 'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),'rng_by_rank':rngs}
        temporary=path.with_suffix('.tmp');torch.save(payload,temporary);temporary.replace(path)
        atomic_json(path.parent/'latest.json',{'checkpoint':path.name,'global_step':state['global_step'],
                                              'completed_epoch':state['completed_epoch']})
    if world>1: dist.barrier()


def train(args):
    import yaml
    from encoder.model import PooledDiSCoTME
    config=yaml.safe_load(Path(args.config).read_text())
    data_cfg=dict(config['data']);training=dict(config.get('training',{}));model_cfg=dict(config['model'])
    if args.batch_size is not None: training['batch_size']=args.batch_size
    if args.epochs is not None: training['epochs']=args.epochs
    negative_mode=training.get('negative_mode','all')
    if negative_mode not in ('all','same_section','count_matched_mixed'): raise ValueError('Invalid negative mode')
    epochs=int(training.get('epochs',10));batch_size=int(training.get('batch_size',1));seed=int(training.get('seed',42))
    if min(epochs,batch_size)<1 or (args.max_steps is not None and args.max_steps<1): raise ValueError('Invalid run length')
    rank,world=rank_world();local_rank=int(os.environ.get('LOCAL_RANK',0))
    device=torch.device('cuda',local_rank) if torch.cuda.is_available() else torch.device('cpu')
    if device.type!='cuda' and bool(training.get('require_cuda',True)):
        raise RuntimeError('Production training requires CUDA; CPU is only an explicit test option')
    random.seed(seed+rank);np.random.seed(seed+rank);torch.manual_seed(seed+rank)
    if device.type=='cuda': torch.cuda.manual_seed_all(seed+rank)
    torch.backends.cudnn.benchmark=False
    root=Path(args.run_dir).resolve();root.mkdir(parents=True,exist_ok=True)
    if (root/'epochs.jsonl').exists() and not args.resume: raise RuntimeError('Existing training output; use a fresh run or explicit resume')
    checkpoint_dir=root/'checkpoints';checkpoint_dir.mkdir(exist_ok=True)
    cohort=CohortDataset(**data_cfg)
    dataset=RGBTrainingDataset(cohort)
    sampler=DistributedBalancedSampler(cohort,batch_size,world,rank,seed,training.get('steps_per_epoch'))
    loader_generator=torch.Generator().manual_seed(seed+rank)
    workers=int(training.get('num_workers',2))
    loader=DataLoader(dataset,batch_size=batch_size,sampler=sampler,collate_fn=collate_training,
                      num_workers=workers,pin_memory=device.type=='cuda',drop_last=True,
                      worker_init_fn=seed_worker,generator=loader_generator,
                      persistent_workers=workers>0,**({'prefetch_factor':1} if workers else {}))
    if 'gene_dim' in model_cfg and int(model_cfg['gene_dim'])!=cohort.gene_dim:
        raise ValueError('Configured gene_dim does not match input gene list')
    model_cfg['gene_dim']=cohort.gene_dim
    model=build_model(model_cfg,cohort.gene_dim,device)
    groups,group_names=optimizer_groups(model,training)
    optimizer=torch.optim.AdamW(groups,weight_decay=float(training.get('weight_decay',1e-5)))
    total_steps=epochs*len(loader)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=max(1,total_steps),eta_min=0.)
    amp=bool(training.get('amp',True)) and device.type=='cuda'
    scaler=torch.amp.GradScaler('cuda',enabled=amp,init_scale=float(training.get('initial_loss_scale',4096.)))
    start_epoch=0;resume_step=0;global_step=0;optimizer_steps=0
    if args.resume:
        checkpoint=torch.load(args.resume,map_location='cpu',weights_only=False)
        if checkpoint.get('probe_checkpoint',False):
            raise ValueError('Probe checkpoints are diagnostic; start the production run separately')
        if checkpoint['world_size']!=world: raise ValueError('Exact resume requires the same world size')
        if checkpoint['input_contract']!=cohort.input_contract: raise ValueError('Resume input contract differs')
        if checkpoint['model_config']!=model_cfg: raise ValueError('Resume model configuration differs')
        previous_training=checkpoint['training_config']
        for field in ['epochs','batch_size','steps_per_epoch','seed','negative_mode']:
            if previous_training.get(field)!=training.get(field): raise ValueError(f'Resume changes {field}')
        model.load_state_dict(checkpoint['model']);optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler']);scaler.load_state_dict(checkpoint['scaler'])
        start_epoch=int(checkpoint['epoch'])+int(checkpoint['completed_epoch'])
        resume_step=0 if checkpoint['completed_epoch'] else int(checkpoint['next_step_in_epoch'])
        global_step=int(checkpoint['global_step']);optimizer_steps=int(checkpoint['optimizer_steps'])
        restore_rng(checkpoint['rng_by_rank'][rank])
    if world>1:
        model=nn.parallel.DistributedDataParallel(model,device_ids=[local_rank] if device.type=='cuda' else None,
                                                  broadcast_buffers=False,find_unused_parameters=True)
    bare=model.module if hasattr(model,'module') else model
    if rank==0 and not args.resume:
        atomic_json(root/'initialization.json', {'model_state_sha256':model_state_sha256(bare.state_dict()),
             'seed':seed,'world_size':world,'source':'same original constructor and DINO initialization; no trained checkpoint'})
    resolved={**config,'data':data_cfg,'model':model_cfg,'training':training,'world_size':world,
              'per_gpu_batch_size':batch_size,'global_batch_size':batch_size*world,'steps_per_epoch':len(loader),
              'epochs':epochs,'max_steps_probe':args.max_steps,'seed':seed,
              'image_normalization':{'mean':[.485,.456,.406],'std':[.229,.224,.225]},
              'loss':'local-anchor/global-key symmetric multi-positive InfoNCE; differentiable all_gather; no world-size multiplier',
              'negative_mode':negative_mode,'negative_subsampling_rng':'separate generator; seed + 1000003 * global_step + rank',
              'sampler':'deterministic global replacement draws weighted inverse sample size; equal rank shards',
              'optimizer_group_parameter_names':group_names,'input_contract':cohort.input_contract}
    if rank==0:
        atomic_json(root/'resolved_config.json',resolved)
        atomic_json(root/'input_contract.json',cohort.input_contract)
        atomic_json(root/'model_provenance.json',bare.provenance())
        (root/'gene_ids.txt').write_text('\n'.join(cohort.gene_ids)+'\n')
    signal=torch.zeros((),dtype=torch.int32,device=device)
    if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
    log_every=max(1,int(training.get('log_every',10)));temperature=float(training.get('temperature',.07))
    clip=float(training.get('grad_clip',1.));consecutive_overflows=0;stop=False;started=time.monotonic()
    graceful_stop={'requested':False}
    def request_stop(signum, frame): graceful_stop['requested']=True
    signal_module.signal(signal_module.SIGTERM,request_stop)
    signal_module.signal(signal_module.SIGUSR1,request_stop)
    max_seconds=args.max_seconds if args.max_seconds is not None else training.get('max_run_seconds')
    stopped_for_resume=False
    for epoch in range(start_epoch,epochs):
        sampler.set_epoch(epoch);sampler.start_step=resume_step if epoch==start_epoch else 0
        step_offset=sampler.start_step;model.train();epoch_losses=[];epoch_started=time.monotonic()
        statistic_keys=['negative_count_mean','within_negative_count_mean','cross_negative_count_mean','positive_count_mean','zero_negative_fraction']
        epoch_statistics=[];draw_hash=hashlib.sha256()
        for step,batch in enumerate(loader):
            step_started=time.monotonic();optimizer.zero_grad(set_to_none=True)
            draw_hash.update(batch['global_spot_index'].numpy().tobytes())
            # Image I/O exceptions terminate the torchrun worker group. Once all
            # ranks enter a step, forward OOM is synchronized BEFORE all_gather.
            output=None;forward_error=None
            try:
                tensors={k:v.to(device,non_blocking=True) for k,v in batch.items() if torch.is_tensor(v)}
                with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=amp):
                    output=model(tensors)
            except (torch.cuda.OutOfMemoryError,RuntimeError) as exc:
                forward_error=exc
            if synchronized_bad(forward_error is not None,signal):
                if forward_error is not None: print(f'RANK {rank} forward failed: {forward_error}',flush=True)
                raise RuntimeError('At least one rank failed during transfer/forward; aborting all ranks before contrastive collectives')
            loss,loss_statistics=distributed_multi_positive_loss(output['image_embedding'],output['gene_embedding'],
                tensors['global_spot_index'],temperature,section_indices=tensors['sample_index'],
                negative_mode=negative_mode,negative_seed=seed+1000003*global_step+rank,return_stats=True)
            if synchronized_bad(not bool(torch.isfinite(loss)),signal):
                raise FloatingPointError('Nonfinite contrastive loss on at least one rank')
            # Backward OOM cannot be safely recovered mid-DDP collective. Raise
            # immediately; torchrun's elastic supervisor terminates peer ranks.
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),clip,error_if_nonfinite=False)
            overflow=synchronized_bad(not bool(torch.isfinite(grad_norm)),signal)
            if overflow:
                if not amp: raise FloatingPointError('Nonfinite gradients without AMP')
                consecutive_overflows+=1;scaler.update(new_scale=max(scaler.get_scale()/2,1e-8))
                if consecutive_overflows>int(training.get('max_consecutive_overflows',20)):
                    raise FloatingPointError('Repeated AMP overflows after dynamic loss-scale reductions')
            else:
                consecutive_overflows=0;scaler.step(optimizer);scaler.update();scheduler.step();optimizer_steps+=1
            global_step+=1
            values=torch.stack([loss.detach().float(),output['alpha_image'].detach().float().mean(),
                                output['alpha_gene'].detach().float().mean()])
            if world>1: dist.all_reduce(values);values/=world
            epoch_losses.append(float(values[0]))
            statistics=torch.stack([loss_statistics[key].detach().float() for key in statistic_keys])
            if world>1: dist.all_reduce(statistics);statistics/=world
            epoch_statistics.append(statistics.cpu().tolist())
            log_step=global_step%log_every==0 or step==0 or args.max_steps is not None
            if log_step:
                memory=torch.tensor([torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0,
                                     torch.cuda.max_memory_reserved(device) if device.type=='cuda' else 0],dtype=torch.float64,device=device)
                if world>1: dist.all_reduce(memory,op=dist.ReduceOp.MAX)
                record={'epoch':epoch+1,'step_in_epoch':step+1+step_offset,'global_step':global_step,'optimizer_steps':optimizer_steps,
                        'loss':float(values[0]),'alpha_image':float(values[1]),'alpha_gene':float(values[2]),
                        'grad_norm':float(grad_norm) if torch.isfinite(grad_norm) else None,'amp_overflow_skipped':overflow,
                        'loss_scale':scaler.get_scale(),'lr':{p['name']:p['lr'] for p in optimizer.param_groups},
                        'peak_allocated_gib_max_rank':float(memory[0]/1024**3),
                        'peak_reserved_gib_max_rank':float(memory[1]/1024**3),
                        'step_seconds':time.monotonic()-step_started,'elapsed_seconds':time.monotonic()-started}
                record.update(dict(zip(statistic_keys,epoch_statistics[-1])))
                record['negative_mode']=negative_mode
                if rank==0:
                    with (root/'train_steps.jsonl').open('a') as f: f.write(json.dumps(record,allow_nan=False)+'\n')
                    print(json.dumps(record,allow_nan=False),flush=True)
            deadline=(max_seconds is not None and time.monotonic()-started>=float(max_seconds))
            stopped_for_resume=synchronized_bad(graceful_stop['requested'] or deadline,signal)
            if stopped_for_resume or (args.max_steps is not None and global_step>=args.max_steps):
                stop=True;break
        sampler_hashes=[None]*world
        if world>1: dist.all_gather_object(sampler_hashes,draw_hash.hexdigest())
        else: sampler_hashes=[draw_hash.hexdigest()]
        completed_epoch=step+1+step_offset==sampler.steps_per_epoch
        state={'epoch':epoch,'completed_epoch':completed_epoch,'next_step_in_epoch':step+1+step_offset,
               'probe_checkpoint':args.max_steps is not None,'config':resolved,'global_step':global_step,'optimizer_steps':optimizer_steps,
               'world_size':world,'model_config':model_cfg,'training_config':training,'input_contract':cohort.input_contract,
               'gene_ids':cohort.gene_ids,'epoch_mean_loss':sum(epoch_losses)/len(epoch_losses)}
        prefix='probe_step' if args.max_steps is not None else 'step'
        filename=f'epoch_{epoch+1:04d}.pt' if completed_epoch else f'{prefix}_{global_step:06d}.pt'
        save_checkpoint(checkpoint_dir/filename,bare,optimizer,scheduler,scaler,state,rank,world)
        if rank==0:
            with (root/'epochs.jsonl').open('a') as f:
                f.write(json.dumps({'epoch':epoch+1,'completed_epoch':completed_epoch,'mean_loss':state['epoch_mean_loss'],
                                    'seconds':time.monotonic()-epoch_started,'checkpoint':filename,
                                    'negative_mode':negative_mode,'sampler_draw_sha256_by_rank':sampler_hashes,
                                    **dict(zip(statistic_keys,np.mean(epoch_statistics,axis=0).tolist()))})+'\n')
        if stop: break
    if rank==0:
        atomic_json(root/'run_status.json',{'status':'probe_complete' if args.max_steps is not None else 'checkpointed_for_resume' if stopped_for_resume else 'training_complete',
                                          'global_step':global_step,'optimizer_steps':optimizer_steps,'epochs_requested':epochs,
                                          'world_size':world,'elapsed_seconds':time.monotonic()-started})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--run-dir',required=True)
    parser.add_argument('--batch-size',type=int);parser.add_argument('--epochs',type=int)
    parser.add_argument('--max-steps',type=int);parser.add_argument('--max-seconds',type=float);parser.add_argument('--resume')
    args=parser.parse_args()
    local_rank=int(os.environ.get('LOCAL_RANK',0));world=int(os.environ.get('WORLD_SIZE',1))
    try:
        if torch.cuda.is_available(): torch.cuda.set_device(local_rank)
        if world>1:
            dist.init_process_group(backend='nccl' if torch.cuda.is_available() else 'gloo',
                                    timeout=timedelta(minutes=5))
        train(args)
    except BaseException as exc:
        rank=int(os.environ.get('RANK',0));error={'status':'failed','rank':rank,'type':type(exc).__name__,
                                               'error':str(exc),'traceback':traceback.format_exc()}
        try: atomic_json(Path(args.run_dir)/f'error_rank_{rank:03d}.json',error)
        except Exception: pass
        print(error['traceback'],file=sys.stderr,flush=True)
        # No barrier here: a rank may have failed during a collective. torchrun
        # must observe a nonzero exit and terminate the remaining worker group.
        raise
    else:
        if dist.is_initialized(): dist.destroy_process_group()


if __name__=='__main__':
    main()
