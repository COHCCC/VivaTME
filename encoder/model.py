"""Pooled DiSCoTME with trainable DINO and the original LongNet spatial encoder.

Adaptations required by cohort inputs: a shared gene axis, relative micrometer
coordinates, compacted/masked neighbors, and a local-only empty-neighborhood
fallback. No dense Transformer substitution or learned motif-matching module.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange
from torchscale.architecture.config import EncoderConfig
from torchscale.component.dilated_attention import DilatedAttention
from .dilated_blocks import (
    DilatedEncoder, DilatedEncoderLayer, compute_2d_sincos_pos_embed,
)


class AuditableDilatedAttention(DilatedAttention):
    """Original sparse gathering/scattering; optional CPU reference kernel.

    ``torchscale`` uses the installed fused backend: V100 -> xformers CUTLASS.
    ``torch_reference`` evaluates the SAME already-gathered sparse streams
    using ordinary torch math, only for small CPU correctness tests.
    """
    def __init__(self, *args, attention_backend="torchscale", **kwargs):
        super().__init__(*args, **kwargs)
        self.attention_backend = attention_backend

    def attention_ops(self, q, k, v, key_padding_mask=None, attn_mask=None,
                      rel_pos=None, is_causal=False):
        if key_padding_mask is not None and bool(key_padding_mask.any()):
            raise ValueError("LongNet must receive compact valid-neighbor sequences, not padded tokens")
        if self.attention_backend == "torchscale":
            if not q.is_cuda:
                raise RuntimeError("Production LongNet requires CUDA; use torch_reference for CPU checks")
            return super().attention_ops(q, k, v, key_padding_mask=None,
                                         attn_mask=attn_mask, rel_pos=rel_pos, is_causal=is_causal)
        if self.attention_backend != "torch_reference":
            raise ValueError(f"Unknown attention backend: {self.attention_backend}")
        if q.is_cuda:
            raise RuntimeError("torch_reference is CPU-audit-only and cannot replace production CUDA LongNet")
        if attn_mask is not None or rel_pos is not None or is_causal:
            raise ValueError("Reference backend is for noncausal spatial attention only")
        # q/k/v have already been dilated by the original LongNet implementation.
        scores = torch.bmm(q.float(), k.float().transpose(1, 2)) * self.scaling
        lse = torch.logsumexp(scores, dim=-1)
        probs = torch.softmax(scores, dim=-1).to(v.dtype)
        probs = F.dropout(probs, p=self.dropout, training=self.training)
        result = torch.bmm(probs, v)
        result = rearrange(result, '(b h) l d -> b l (h d)', h=self.num_heads)
        lse = rearrange(lse, '(b h) l -> b h l', h=self.num_heads)
        return result, lse


class PooledDilatedEncoderLayer(DilatedEncoderLayer):
    def build_self_attention(self, embed_dim, args):
        return AuditableDilatedAttention(args, embed_dim, args.encoder_attention_heads,
            dropout=args.attention_dropout, self_attention=True, encoder_decoder_attention=False,
            subln=args.subln, attention_backend=args.crosmotme_attention_backend)


class PooledDilatedEncoder(DilatedEncoder):
    def build_encoder_layer(self, args, depth, is_moe_layer=False, is_encoder_decoder=False):
        return PooledDilatedEncoderLayer(args, depth, is_moe_layer=is_moe_layer,
                                         is_encoder_decoder=is_encoder_decoder)


def build_longnet(dim, heads, layers, dropout, attention_backend):
    if attention_backend not in {"torchscale", "torch_reference"}:
        raise ValueError("attention_backend must be torchscale or torch_reference")
    if dim % heads or dim % 4 or layers < 1:
        raise ValueError("Invalid LongNet dimensions")
    args = EncoderConfig(encoder_layers=layers, encoder_embed_dim=dim,
        encoder_ffn_embed_dim=4 * dim, encoder_attention_heads=heads,
        dilated_ratio='[1, 2, 3, 4]', segment_length='[1000, 2000, 3000, 5000]',
        block_shift=True, flash_attention=True, dropout=dropout, drop_path_rate=dropout,
        attention_dropout=0., use_xmoe=False, moe_top1_expert=False, moe_freq=0,
        moe_expert_count=0, seq_parallel=False, checkpoint_activations=False, fsdp=False)
    args.crosmotme_attention_backend = attention_backend
    return PooledDilatedEncoder(args)


class FidelityGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(dim, 128), nn.GELU(), nn.Linear(128, 1), nn.Sigmoid())

    def forward(self, local, context):
        alpha = self.gate(local)
        return alpha * local + (1 - alpha) * context, alpha


class LongNetContext(nn.Module):
    """Neighbor-only LongNet; padding is removed before any attention kernel."""
    def __init__(self, input_dim, embed_dim, layers, heads, dropout,
                 position_scale_um, attention_backend):
        super().__init__()
        if position_scale_um <= 0:
            raise ValueError("position_scale_um must be positive")
        self.position_scale_um = float(position_scale_um)
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.dilated_transformer = build_longnet(embed_dim, heads, layers, dropout, attention_backend)
        self.fidelity_gate = FidelityGate(embed_dim)

    def forward(self, target, neighbors, relative_pos_um, mask):
        if mask.dtype != torch.bool or neighbors.shape[:2] != mask.shape:
            raise ValueError("Neighbor features require a Boolean [B,K] mask")
        if relative_pos_um.shape != mask.shape + (2,):
            raise ValueError("Relative positions must be [B,K,2]")
        local = self.input_proj(target)
        lengths = mask.sum(1)
        context = torch.zeros_like(local)
        # Padding may be NaN; it is never projected, encoded, or pooled.
        # Valid-neighbor order is preserved inside each length bucket.
        for length in torch.unique(lengths).detach().cpu().tolist():
            if length == 0:
                continue
            rows = torch.nonzero(lengths == length, as_tuple=False).flatten()
            selected = mask[rows]
            valid_neighbors = neighbors[rows][selected].reshape(len(rows), length, -1)
            positions = relative_pos_um[rows][selected].reshape(len(rows), length, 2)
            projected = self.input_proj(valid_neighbors)
            pe = compute_2d_sincos_pos_embed(positions.float() / self.position_scale_um,
                                            local.shape[-1], device=local.device).to(projected.dtype)
            tokens = projected + .3 * pe
            result = self.dilated_transformer(src_tokens=None, token_embeddings=tokens)['encoder_out']
            context[rows] = result.float().mean(dim=1).to(context.dtype)
        fused, alpha = self.fidelity_gate(local, context)
        has_context = lengths > 0
        # Use the projected local identity exactly if there is no valid neighbor.
        fused = torch.where(has_context[:, None], fused, local)
        alpha = torch.where(has_context[:, None], alpha, torch.ones_like(alpha))
        return fused, alpha


class ResidualGeneEncoder(nn.Module):
    """The release gene_mlp_residual identity branch (D -> 512 residual)."""
    def __init__(self, gene_dim, dropout=.1):
        super().__init__()
        self.input_norm = nn.LayerNorm(gene_dim)
        self.input_proj = nn.Linear(gene_dim, 512)
        self.mlp = nn.Sequential(nn.Linear(512, 512), nn.LayerNorm(512), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(512, 512), nn.LayerNorm(512))

    def forward(self, x):
        projected = self.input_proj(self.input_norm(x))
        return projected + self.mlp(projected)


class PooledDiSCoTME(nn.Module):
    """Shared end-to-end DiSCoTME backbone, initialized from local DINO weights.

    Images must already be RGB 224x224 tensors with ImageNet normalization.
    This class does not load files, sample neighbors, normalize genes or resize.
    All DINO parameters are trainable. All samples use the SAME model instance.
    """
    def __init__(self, gene_dim, checkpoint_path, embed_dim=256, proj_dim=128,
                 num_context_layers=4, num_heads=8, dropout=.1,
                 position_scale_um=200., gradient_checkpointing=False,
                 attention_backend="torchscale"):
        super().__init__()
        if gene_dim < 1 or embed_dim < 1 or proj_dim < 1:
            raise ValueError("Feature dimensions must be positive")
        from safetensors.torch import load_file
        import timm
        checkpoint = Path(checkpoint_path).expanduser().resolve(strict=True)
        if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
            raise ValueError("Checkpoint must be a nonempty local safetensors file")
        digest = hashlib.sha256()
        with checkpoint.open('rb') as f:
            for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
                digest.update(chunk)
        self.checkpoint_sha256 = digest.hexdigest()
        self.checkpoint_path = str(Path(checkpoint_path).expanduser().absolute())
        self.gene_dim = int(gene_dim)
        self.backbone = timm.create_model('vit_small_patch16_224.dino', pretrained=False, num_classes=0)
        self.backbone.load_state_dict(load_file(str(checkpoint), device='cpu'), strict=True)
        if self.backbone.num_features != 384:
            raise ValueError("Expected original DINO ViT-S/16 with 384-dimensional CLS features")
        self.backbone.requires_grad_(True)
        if gradient_checkpointing:
            self.backbone.set_grad_checkpointing(enable=True)
        self.image_proj = nn.Linear(384, embed_dim)
        context_args = dict(embed_dim=embed_dim, layers=num_context_layers, heads=num_heads,
            dropout=dropout, position_scale_um=position_scale_um, attention_backend=attention_backend)
        self.image_context = LongNetContext(embed_dim, **context_args)
        self.gene_encoder = ResidualGeneEncoder(gene_dim, dropout)
        self.gene_context = LongNetContext(512, **context_args)
        self.img_head = nn.Sequential(nn.Linear(embed_dim, proj_dim), nn.ReLU(), nn.Linear(proj_dim, proj_dim))
        self.gene_head = nn.Sequential(nn.Linear(embed_dim, proj_dim), nn.ReLU(), nn.Linear(proj_dim, proj_dim))
        self.architecture_config = dict(gene_dim=gene_dim, image_backbone='vit_small_patch16_224.dino',
            image_backbone_trainable=True, embed_dim=embed_dim, proj_dim=proj_dim,
            num_context_layers=num_context_layers, num_heads=num_heads, dropout=dropout,
            position_scale_um=position_scale_um, gradient_checkpointing=gradient_checkpointing,
            attention_backend=attention_backend, dilated_ratio=[1,2,3,4],
            segment_length=[1000,2000,3000,5000], seq_parallel=False,
            neighbor_mask_strategy='compact_valid_neighbors_and_bucket_by_length',
            empty_neighborhood='context_input_projection_of_local_identity',
            joint_embedding='arithmetic_mean_of_raw_image_and_gene_projection_heads',
            internal_dilation_padding='original_torchscale_semantics_preserved')
        self._timm_version = timm.__version__

    def provenance(self):
        return dict(architecture=self.architecture_config, checkpoint_path=self.checkpoint_path,
            checkpoint_sha256=self.checkpoint_sha256, torch_version=torch.__version__,
            timm_version=self._timm_version)

    def forward(self, batch):
        images, neighbor_images = batch['target_img'], batch['context_imgs']
        genes, neighbor_genes = batch['target_gene'], batch['context_genes']
        positions, mask = batch['relative_pos_um'], batch['neighbor_mask']
        if mask.dtype != torch.bool or mask.ndim != 2:
            raise ValueError("neighbor_mask must be Boolean [B,K]")
        b, k = mask.shape
        if images.shape != (b, 3, 224, 224) or neighbor_images.shape != (b, k, 3, 224, 224):
            raise ValueError("Image tensors must be [B,3,224,224] / [B,K,3,224,224]")
        if genes.shape != (b, self.gene_dim) or neighbor_genes.shape != (b, k, self.gene_dim):
            raise ValueError("Gene tensors do not match the shared input dimension")
        # Encode only valid entries: padded values cannot enter DINO/MLP autograd.
        valid_images = torch.cat((images, neighbor_images[mask]), dim=0)
        image_features = self.image_proj(self.backbone(valid_images))
        image_target = image_features[:b]
        image_neighbors = image_features.new_zeros((b, k, image_features.shape[-1]))
        image_neighbors[mask] = image_features[b:]
        valid_genes = torch.cat((genes, neighbor_genes[mask]), dim=0)
        gene_features = self.gene_encoder(valid_genes)
        gene_target = gene_features[:b]
        gene_neighbors = gene_features.new_zeros((b, k, 512))
        gene_neighbors[mask] = gene_features[b:]
        image_fused, alpha_image = self.image_context(image_target, image_neighbors, positions, mask)
        gene_fused, alpha_gene = self.gene_context(gene_target, gene_neighbors, positions, mask)
        zi, zg = self.img_head(image_fused), self.gene_head(gene_fused)
        return dict(image_embedding=zi, gene_embedding=zg, alpha_image=alpha_image,
                    alpha_gene=alpha_gene, joint_embedding=(zi+zg)*.5,
                    z_img=zi, z_gene=zg, alpha_img=alpha_image)
