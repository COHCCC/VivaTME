"""Controlled negative-scope variants of pooled DiSCoTME's paired InfoNCE.

All modes retain local anchors, differentiable globally gathered keys, repeated
spot multi-positives, and the same denominator mask in both modality directions.
The caller supplies a per-step/per-rank negative_seed; this module does not add
rank and never advances the default CPU/CUDA random generator.
"""
from __future__ import annotations

import math
import operator
import torch
import torch.distributed as dist
from torch.nn import functional as F

_MODES = ('all', 'same_section', 'count_matched_mixed')
_INTEGER_DTYPES = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)


def _negative_key_mask(positive, same_section, negative_mode, negative_seed):
    """Return selected nonpositive KEY ENTRIES, without replacement per anchor.

    For count matching, independent iid float64 random priorities give a
    uniform subset of nonpositive entries. Selection uses no embedding values.
    Positive duplicate entries are all excluded from the sampling pool.
    """
    candidates = ~positive
    if negative_mode == 'all':
        return candidates
    if negative_mode == 'same_section':
        return candidates & same_section
    counts = (candidates & same_section).sum(dim=1)
    chosen = torch.zeros_like(positive)
    maximum = int(counts.max().item())
    if maximum == 0:
        return chosen
    generator = torch.Generator(device=positive.device)
    generator.manual_seed(operator.index(negative_seed))
    priorities = torch.rand(positive.shape, device=positive.device,
                            dtype=torch.float64, generator=generator)
    priorities.masked_fill_(positive, -torch.inf)
    positions = priorities.topk(maximum, dim=1, largest=True, sorted=True).indices
    keep = torch.arange(maximum, device=positive.device)[None, :] < counts[:, None]
    chosen.scatter_(1, positions, keep)
    return chosen


def distributed_multi_positive_loss(image, gene, spot_indices, temperature=.07,
                                    section_indices=None, negative_mode='all',
                                    negative_seed=0, return_stats=False):
    """Symmetric InfoNCE; DDP must provide equal local batch sizes on all ranks.

    ``all`` is numerically identical to the legacy implementation. Optional
    section_indices do not change this mode's denominator, but permit section
    consistency validation and within/cross-section negative diagnostics.

    ``same_section`` removes nonpositive keys from other sections.
    ``count_matched_mixed`` selects, for each anchor, as many arbitrary global
    nonpositive key entries as same_section would retain. All positive entries
    remain in numerator and denominator in every mode. Duplicate sampled spots
    count as entries, not distinct biological spots.

    A zero-negative anchor contributes exactly zero and remains in the local
    batch mean. No world-size multiplier is applied: differentiable all_gather
    sums key-side gradients; DDP subsequently averages parameter gradients.

    With return_stats, return (loss, detached scalar-tensor local means):
    negative_count_mean, within_negative_count_mean, cross_negative_count_mean,
    positive_count_mean, zero_negative_fraction. Without section_indices the
    within/cross counts are NaN, since section identity is not available.
    """
    if image.ndim != 2 or image.shape != gene.shape or len(image) < 1:
        raise ValueError('Invalid paired embedding shapes')
    if not math.isfinite(float(temperature)) or temperature <= 0:
        raise ValueError('Temperature must be finite and positive')
    if image.device != gene.device:
        raise ValueError('Paired embeddings must be on the same device')
    if spot_indices.shape != (len(image),) or spot_indices.dtype not in _INTEGER_DTYPES:
        raise ValueError('Spot indices must be an integer vector matching the batch')
    if spot_indices.device != image.device:
        raise ValueError('Spot indices and embeddings must be on the same device')
    if negative_mode not in _MODES:
        raise ValueError(f'Unknown negative_mode: {negative_mode}')
    if negative_mode != 'all' and section_indices is None:
        raise ValueError('section_indices are required for scoped negatives')
    if section_indices is not None:
        if section_indices.shape != spot_indices.shape or section_indices.dtype not in _INTEGER_DTYPES:
            raise ValueError('Section indices must be an integer vector matching the batch')
        if section_indices.device != image.device:
            raise ValueError('Section indices and embeddings must be on the same device')
    if negative_mode == 'count_matched_mixed':
        operator.index(negative_seed)  # Fail clearly for non-integral seeds.

    # Keep the legacy normalization and gather order for controlled equivalence.
    image, gene = F.normalize(image.float(), dim=-1), F.normalize(gene.float(), dim=-1)
    world = dist.get_world_size() if dist.is_initialized() else 1
    if world > 1:
        from torch.distributed.nn.functional import all_gather
        all_image = torch.cat(all_gather(image), dim=0)
        all_gene = torch.cat(all_gather(gene), dim=0)
        index_parts = [torch.empty_like(spot_indices) for _ in range(world)]
        dist.all_gather(index_parts, spot_indices)
        all_indices = torch.cat(index_parts)
        if section_indices is not None:
            section_parts = [torch.empty_like(section_indices) for _ in range(world)]
            dist.all_gather(section_parts, section_indices)
            all_sections = torch.cat(section_parts)
        else:
            all_sections = None
    else:
        all_image, all_gene, all_indices = image, gene, spot_indices
        all_sections = section_indices

    positive = spot_indices[:, None].eq(all_indices[None, :])
    if not positive.any(dim=1).all():
        raise RuntimeError('An anchor has no positive key')
    if all_sections is not None:
        # Every rank sees the same full mapping, including duplicate spots on
        # different ranks, so all ranks reject a conflicting mapping together.
        order = torch.argsort(all_indices)
        ordered_spots, ordered_sections = all_indices[order], all_sections[order]
        inconsistent = (ordered_spots[1:] == ordered_spots[:-1]) & (ordered_sections[1:] != ordered_sections[:-1])
        if inconsistent.any():
            raise ValueError('A global spot index maps to multiple section indices')
        same_section = section_indices[:, None].eq(all_sections[None, :])
    else:
        same_section = None

    negative = _negative_key_mask(positive, same_section, negative_mode, negative_seed)
    denominator = positive | negative
    has_negative = negative.any(dim=1)

    def directional(anchors, keys):
        scores = anchors @ keys.T / temperature
        if negative_mode == 'all':
            # Exact legacy expression; do not insert a masking transform here.
            return (torch.logsumexp(scores, dim=1) - torch.logsumexp(
                scores.masked_fill(~positive, -torch.inf), dim=1)).mean()
        row_loss = torch.logsumexp(scores.masked_fill(~denominator, -torch.inf), dim=1) - torch.logsumexp(
            scores.masked_fill(~positive, -torch.inf), dim=1)
        return row_loss.masked_fill(~has_negative, 0.).mean()

    loss = .5 * (directional(image, all_gene) + directional(gene, all_image))
    if not return_stats:
        return loss
    stats = {
        'negative_count_mean': negative.sum(dim=1).float().mean().detach(),
        'positive_count_mean': positive.sum(dim=1).float().mean().detach(),
        'zero_negative_fraction': (~has_negative).float().mean().detach(),
    }
    if same_section is None:
        stats['within_negative_count_mean'] = loss.detach().new_tensor(float('nan'))
        stats['cross_negative_count_mean'] = loss.detach().new_tensor(float('nan'))
    else:
        stats['within_negative_count_mean'] = (negative & same_section).sum(dim=1).float().mean().detach()
        stats['cross_negative_count_mean'] = (negative & ~same_section).sum(dim=1).float().mean().detach()
    return loss, stats
