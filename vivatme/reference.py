"""Initial reference fit on U (before alignment).

1. shared-compartment model (Methods 3.3), 10 k-means++ restarts;
2. spatial blocks within each (section, compartment) and a fixed-assignment block-permutation threshold for the
   reference's own deviation fit (used only to initialise training and the read-out; the reported threshold comes
   from the full-refit null, see null.py);
3. joint compartment + deviation fit at that threshold. Its labels initialise adapter training and define the
   cross-section matching graph; its blocks define the units permuted by the full-refit null.
"""
import time
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits
from .compartments import fit_compartments_multi, full_q
from .deviation import sufficient, fit_joint

BLOCK_SEED = 20260924
BLOCK_TARGET = 25        # spots per spatial block
BLOCK_DIAMETER = 550.    # micrometres
N_PERM = 500


def _merge_tiny(labels, xy):
    labels = labels.copy()
    while True:
        unique, count = np.unique(labels, return_counts=True)
        small = unique[count < 3]
        if not len(small) or len(unique) == 1:
            break
        source = int(small[0]); other = unique[unique != source]
        source_mean = xy[labels == source].mean(0)
        target = int(other[np.argmin([np.sum((xy[labels == j].mean(0) - source_mean) ** 2) for j in other])])
        labels[labels == source] = target
    return np.unique(labels, return_inverse=True)[1]


def spatial_blocks(meta, labels, k):
    """Contiguous blocks of ~25 spots (diameter <= 550 um) within each (section, compartment); -1 for dustbin spots."""
    sections = meta.sample_index.to_numpy(int); coords = meta[['x_um', 'y_um']].to_numpy(float)
    block_id = np.full(len(meta), -1, int); rows = []
    for s in range(int(sections.max()) + 1):
        for kk in range(k):
            ids = np.flatnonzero((sections == s) & (labels == kk))
            if not len(ids):
                continue
            count = max(1, int(np.ceil(len(ids) / BLOCK_TARGET)))
            local = KMeans(n_clusters=count, n_init=10, random_state=BLOCK_SEED + s * 100 + kk).fit_predict(coords[ids])
            leaves = []

            def split_block(index, depth=0):
                diameter = float(pdist(coords[ids[index]]).max()) if len(index) > 1 else 0.
                if diameter <= BLOCK_DIAMETER or len(index) < 6 or depth >= 12:
                    leaves.append(index); return
                part = KMeans(n_clusters=2, n_init=10, random_state=BLOCK_SEED + s * 100 + kk + depth).fit_predict(coords[ids[index]])
                if len(np.unique(part)) < 2:
                    leaves.append(index); return
                for value in (0, 1):
                    split_block(index[part == value], depth + 1)
            for value in np.unique(local):
                split_block(np.flatnonzero(local == value))
            refined = np.empty(len(ids), int)
            for value, index in enumerate(leaves):
                refined[index] = value
            local = _merge_tiny(refined, coords[ids])
            for j in np.unique(local):
                members = ids[local == j]; block_id[members] = len(rows)
                rows.append(dict(block=len(rows), section=s, prototype=kk, n_spots=len(members)))
    return block_id, pd.DataFrame(rows)


def fixed_assignment_threshold(x, meta, fit, k, n_perm=N_PERM):
    """tau95 of max z under block permutations with the assignments of `fit` held fixed (reference fit only)."""
    sections = meta.sample_index.to_numpy(int); NS = int(sections.max()) + 1
    p = full_q(fit)[:, :-1]; mu = fit['centers_full']
    y = np.asarray(x) - fit['offsets'][sections]; labels = fit['labels']
    block_id, blocks = spatial_blocks(meta, labels, k)
    block_mass, block_residual = [], []
    for b in range(len(blocks)):
        members = np.flatnonzero(block_id == b); mass = p[members].sum(0)
        block_mass.append(mass); block_residual.append(p[members].T @ y[members] - mass[:, None] * mu)
    bm_ = np.stack(block_mass); br = np.stack(block_residual)
    dust = labels < 0
    fixed_mass, fixed_sum = sufficient(y[dust], sections[dust], p[dust], NS)
    fixed_resid = fixed_sum - fixed_mass[:, :, None] * mu

    def stats(dest):
        mass = fixed_mass.copy(); resid = fixed_resid.copy()
        for s in range(NS):
            chosen = dest == s
            mass[s] += bm_[chosen].sum(0); resid[s] += br[chosen].sum(0)
        z = np.divide(np.linalg.norm(resid, axis=2), np.sqrt(mass), out=np.zeros_like(mass), where=mass > 0)
        return mass, z
    original = blocks.section.to_numpy(int); hard = blocks.prototype.to_numpy(int)
    rng = np.random.default_rng(BLOCK_SEED); maxima = []
    for _ in range(n_perm):
        dest = original.copy()
        for kk in range(k):
            ix = np.flatnonzero(hard == kk); dest[ix] = rng.permutation(original[ix])
        mass, z = stats(dest); eligible = mass >= 30
        maxima.append(float(z[eligible].max()))
    maxima = np.array(maxima)
    tau = {str(q): float(np.quantile(maxima, q / 100)) for q in (90, 95, 99)}
    return tau, block_id, blocks, maxima


def fit_reference(x, meta, k, seed=42, threads=3, n_perm=N_PERM, log=print):
    """Stages 1-3 above. Returns (reference fit, compartment fit, its stats, tau dict, spot block ids)."""
    t0 = time.monotonic(); s = meta.sample_index.to_numpy(int)
    with threadpool_limits(threads):
        warm, warm_stats = fit_compartments_multi(x, s, k, seed)
    log(f'compartment fit: restart {warm_stats["selected_initialization"]} cost {warm_stats["expected_cost_final"]:.6f} '
        f'converged {warm_stats["converged"]} ({time.monotonic() - t0:.0f}s)')
    with threadpool_limits(threads):
        tau, block_id, blocks, maxima = fixed_assignment_threshold(x, meta, warm, k, n_perm)
    log(f'blocks {len(blocks)}, reference tau95 {tau["95"]:.4f} ({time.monotonic() - t0:.0f}s)')
    with threadpool_limits(threads):
        ref, ref_stats = fit_joint(x, s, warm, warm_stats, tau['95'])
    log(f'reference fit: converged {ref_stats["converged"]}, dustbin {ref_stats["dustbin_fraction"]:.3f} ({time.monotonic() - t0:.0f}s)')
    return ref, warm, warm_stats, tau, block_id
