# Copied unchanged from DiSCoTME (https://github.com/COHCCC/DiSCoTME, src/models/dilated_blocks.py, MIT License).
# src/models/dilated_blocks.py

import os, sys
import torch
import torch.nn as nn
import math

from torchscale.architecture.config import EncoderConfig
from torchscale.architecture.encoder import Encoder, EncoderLayer
from torchscale.component.dilated_attention import DilatedAttention
from fairscale.nn import checkpoint_wrapper, wrap

# ------------------------------
# (1) 2D Position Embedding
# ------------------------------
def compute_2d_sincos_pos_embed(positions, embed_dim=256, device=None):
    """
    Generates 2D position embeddings following the standard ViT/Transformer formula:
      PE(pos, 2i)   = sin( pos / 10000^(2i/d) )
      PE(pos, 2i+1) = cos( pos / 10000^(2i/d) )

    Args:
      positions: [batch_size, 2] or [batch_size, n_spots, 2]
    """
    if len(positions.shape) == 3:
        # [B, N, 2]
        B, N, _ = positions.shape
        positions = positions.reshape(-1, 2)  # [B*N, 2]
        reshape_back = True
    else:
        # [B, 2]
        B = positions.shape[0]
        N = 1
        reshape_back = False

    device = device or positions.device

    # Output size: [B*N, embed_dim]
    half_dim = embed_dim // 2
    row_vals = positions[:, 0]  # [B*N]
    col_vals = positions[:, 1]  # [B*N]

    # Apply 1D sin-cos encoding to row and col separately
    row_emb = _get_1d_sin_cos_vit(row_vals, half_dim, device)
    col_emb = _get_1d_sin_cos_vit(col_vals, half_dim, device)

    # Concatenate => [B*N, embed_dim]
    pos_emb = torch.cat([row_emb, col_emb], dim=1)

    if reshape_back:
        pos_emb = pos_emb.reshape(B, N, embed_dim)

    return pos_emb


def _get_1d_sin_cos_vit(pos, dim, device=None):
    assert dim % 2 == 0, "dim must be even"
    half = dim // 2
    i = torch.arange(half, device=device).float()
    exponent = 2 * i / float(dim)
    freq = 1.0 / (10000 ** exponent)

    pos_expand = pos.unsqueeze(1)
    freq_expand = freq.unsqueeze(0)
    angle = pos_expand * freq_expand

    sin_part = torch.sin(angle)
    cos_part = torch.cos(angle)

    emb = torch.cat([sin_part, cos_part], dim=1)
    return emb

# ------------------------------
# (2) Config Dicts
# ------------------------------

DilatedConfig_8layers_256dim = {
    'encoder_layers': 8,
    'encoder_embed_dim': 256,
    'encoder_ffn_embed_dim': 1024,
    'encoder_attention_heads': 16,
    'dilated_ratio': '[1, 2, 4, 8, 16]',
    'segment_length': '[1024, 2048, 4096, 8192, 16384]',
    'block_shift': True,
    'flash_attention': True,
    'use_xmoe': False,
    'moe_top1_expert': False,
    'moe_freq': 0,
    'moe_expert_count': 0
}

# Visium Standard
DilatedConfig_Spots = {
    'encoder_layers': 4,  
    'encoder_embed_dim': 256,
    'encoder_ffn_embed_dim': 1024,
    'encoder_attention_heads': 8, 
    'dilated_ratio': '[1, 2, 3, 4]', 
    'segment_length': '[1000, 2000, 3000, 5000]', 
    'block_shift': True,
    'flash_attention': True,
    'use_xmoe': False,
    'moe_top1_expert': False,
    'moe_freq': 0,
    'moe_expert_count': 0
}

# Visium HD
DilatedConfig_LargeSpatial = {
    'encoder_layers': 8,
    'encoder_embed_dim': 256,
    'encoder_ffn_embed_dim': 1024,
    'encoder_attention_heads': 16,
    'dilated_ratio': '[1, 2, 4, 8, 16, 32]',  
    'segment_length': '[5000, 20000, 50000, 100000, 170000]',  
    'block_shift': True,
    'flash_attention': True,
    'use_xmoe': False,
    'moe_top1_expert': False,
    'moe_freq': 0,
    'moe_expert_count': 0
}

# Default / TME Aware
DilatedConfig_Spatial = {
    'encoder_layers': 4,  
    'encoder_embed_dim': 256,
    'encoder_ffn_embed_dim': 1024,
    'encoder_attention_heads': 8,  
    'dilated_ratio': '[1, 2, 3, 4]',  
    'segment_length': '[1000, 2000, 3000, 5000]',  
    'block_shift': True,
    'flash_attention': True,
    'use_xmoe': False,
    'moe_top1_expert': False,
    'moe_freq': 0,
    'moe_expert_count': 0
}

# Registry map for configs
DilatedConfigs = {
    "LongNet_8_layers_256_dim": DilatedConfig_8layers_256dim,
    "LongNet_for_spots": DilatedConfig_Spots,
    "LongNet_for_large_spatial": DilatedConfig_LargeSpatial,
    "LongNet_for_spatial": DilatedConfig_Spatial 
}

# ------------------------------
# (3) Dilated Modules
# ------------------------------

class DilatedEncoderLayer(EncoderLayer):
    def build_self_attention(self, embed_dim, args):
        return DilatedAttention(
            args,
            embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
            encoder_decoder_attention=False,
            subln=args.subln,
        )

class DilatedEncoder(Encoder):
    def build_encoder_layer(
        self, args, depth, is_moe_layer=False, is_encoder_decoder=False
    ):
        layer = DilatedEncoderLayer(
            args,
            depth,
            is_moe_layer=is_moe_layer,
            is_encoder_decoder=is_encoder_decoder,
        )
        if args.checkpoint_activations:
            layer = checkpoint_wrapper(layer)
        if args.fsdp:
            layer = wrap(layer)
        return layer

# ------------------------------
# (4) Builder Function
# ------------------------------
def build_dilated_transformer(config_name: str,
                              dilated_ratio: str='[1, 2, 3, 4]',  
                              segment_length: str='[1000, 2000, 3000, 5000]',
                              drop_path_rate: float=0.1,
                              dropout: float=0.1):
    """
    Factory function to build the DilatedTransformer (LongNet)
    """
    if config_name not in DilatedConfigs:
        raise ValueError(f"Unknown config_name: {config_name}, please define in DilatedConfigs dict")

    args = DilatedConfigs[config_name].copy()  

    args['dropout'] = dropout
    args['drop_path_rate'] = drop_path_rate
    args['dilated_ratio'] = dilated_ratio
    args['segment_length'] = segment_length

    arch = EncoderConfig(**args)

    model = DilatedEncoder(arch)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Create DilatedTransformer: {config_name}, # params= {n_params/1e6:.2f}M")
    return model