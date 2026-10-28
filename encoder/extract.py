"""Export VivaTME's input U from a trained pooled DiSCoTME checkpoint.

For every spot, F_img and F_gene are the outputs of the image and gene context modules (256-d each, after the
adaptive local/neighbourhood gate, before the projection heads). U = [F_img/||F_img||, F_gene/||F_gene||] (512-d).
Inference uses batch 16 and float16 autocast, as for the published runs. Run on one GPU or with torchrun
(spots are split over ranks; rank 0 merges).

usage: [torchrun --nproc-per-node N] python encoder/extract.py --run-dir RUN [--checkpoint epoch_0010.pt] --out OUT
writes OUT/U.npy, OUT/spots.csv (global_index, spot_id, sample_id, sample_index, x_um, y_um, barcode), OUT/intermediates.npz
"""
import argparse
import json
import os
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from encoder.dataset import CohortDataset, RGBTrainingDataset, collate_training  # noqa: E402
from encoder.train import build_model  # noqa: E402


class ContextCapture:
    """Forward hooks recording the context-module outputs F (fused local/neighbourhood features)."""
    def __init__(self, model):
        self.values = {}
        self.handles = [getattr(model, m + '_context').register_forward_hook(self._hook(m)) for m in ('image', 'gene')]

    def _hook(self, modality):
        def hook(module, inputs, output):
            self.values[modality] = output[0].detach().float()
        return hook


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', required=True); p.add_argument('--checkpoint', default=None); p.add_argument('--out', required=True)
    p.add_argument('--batch-size', type=int, default=16); p.add_argument('--num-workers', type=int, default=2)
    p.add_argument('--sample-ids', default=None, help='comma-separated subset of sections (neighbourhoods are within-section, so '
                   'their features equal those of a full-cohort export)')
    a = p.parse_args()
    run, out = Path(a.run_dir), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    world = int(os.environ.get('WORLD_SIZE', 1)); local = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local); device = torch.device('cuda', local)
    if world > 1:
        dist.init_process_group('nccl', timeout=timedelta(minutes=30))
    rank = dist.get_rank() if world > 1 else 0
    name = a.checkpoint or json.loads((run / 'checkpoints/latest.json').read_text())['checkpoint']
    ckpt = torch.load(run / 'checkpoints' / name, map_location='cpu', weights_only=False)
    data = dict(ckpt['config']['data'])
    if a.sample_ids:
        data['sample_ids'] = a.sample_ids.split(',')
    cohort = CohortDataset(**data)
    if not a.sample_ids and cohort.input_contract != ckpt['input_contract']:
        raise ValueError('Prepared inputs differ from those the checkpoint was trained on')
    if cohort.input_contract['gene_ids_sha256'] != ckpt['input_contract']['gene_ids_sha256']:
        raise ValueError('Gene axis differs from the checkpoint')
    model = build_model(ckpt['model_config'], cohort.gene_dim, device)
    model.load_state_dict(ckpt['model'], strict=True); del ckpt
    model.eval().requires_grad_(False); cap = ContextCapture(model)
    indices = list(range(rank, len(cohort), world))
    loader = DataLoader(RGBTrainingDataset(cohort), batch_size=a.batch_size, sampler=indices, collate_fn=collate_training,
                        num_workers=a.num_workers, pin_memory=True, drop_last=False)
    gi, fi, fg = [], [], []
    with torch.inference_mode():
        for batch in loader:
            tensors = {k: v.to(device, non_blocking=True) for k, v in batch.items() if torch.is_tensor(v)}
            with torch.autocast('cuda', dtype=torch.float16):
                model(tensors)
            gi.append(batch['global_spot_index'].numpy()); fi.append(cap.values['image'].cpu().numpy()); fg.append(cap.values['gene'].cpu().numpy())
    np.savez(out / f'shard_{rank:03d}.npz', global_index=np.concatenate(gi), image_F=np.concatenate(fi), gene_F=np.concatenate(fg))
    if world > 1:
        dist.barrier()
    if rank == 0:
        parts = [np.load(out / f'shard_{r:03d}.npz') for r in range(world)]
        g = np.concatenate([q['global_index'] for q in parts]); order = np.argsort(g, kind='stable')
        assert np.array_equal(g[order], np.arange(len(cohort)))
        image_F = np.concatenate([q['image_F'] for q in parts])[order]; gene_F = np.concatenate([q['gene_F'] for q in parts])[order]
        np.savez(out / 'intermediates.npz', image_F=image_F, gene_F=gene_F)
        a_, b_ = image_F.astype(np.float64), gene_F.astype(np.float64)
        u = np.concatenate([a_ / np.linalg.norm(a_, axis=1, keepdims=True), b_ / np.linalg.norm(b_, axis=1, keepdims=True)], axis=1)
        assert np.isfinite(u).all()
        np.save(out / 'U.npy', u)
        rows = []
        for i, spot_id in enumerate(cohort.spot_ids):
            si = int(cohort.sample_indices[i]); xy = cohort._samples[si]['coordinates'][i - cohort.sample_offsets[si]]
            rows.append(dict(global_index=i, spot_id=spot_id, sample_id=cohort.sample_ids[si], sample_index=si,
                             x_um=xy[0], y_um=xy[1], barcode=spot_id.split('::', 1)[1]))
        pd.DataFrame(rows).to_csv(out / 'spots.csv', index=False)
        for r in range(world):
            (out / f'shard_{r:03d}.npz').unlink()
        print('U', u.shape, '->', out, flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
