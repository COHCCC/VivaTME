"""Full-refit permutation null for the deviation threshold (Methods 3.5 'Calibration').

Units are the spatial blocks of the reference fit split by reference compartment (dustbin spots and spots without a
block are singleton units). Within each reference compartment the units' section labels are permuted, the
representation is moved to the destination section by x' = u - o_ref[s] + o_ref[s'], and the whole pipeline
(reference fit, graphs, adapter training, read-out, fixed-assignment fit with tau = inf) is rerun. The statistic is the
maximum z over eligible cells; tau95 is the 95th percentile of the maxima of 500 permutations.
"""
import numpy as np

SEED_BASE = 20261001


def units(sections, ref_labels, block_id, k):
    single = (ref_labels < 0) | (block_id < 0)
    key = np.where(single, -1, block_id * (k + 1) + ref_labels + 1)
    key[single] = key.max() + 1 + np.arange(single.sum())
    ub, inv = np.unique(key, return_inverse=True)
    usec = np.zeros(len(ub), int); usec[inv] = sections
    ugrp = np.full(len(ub), -1); ugrp[inv] = ref_labels
    assert np.array_equal(usec[inv], sections) and np.array_equal(ugrp[inv], ref_labels)
    return inv, usec, ugrp


def permuted_cohort(x, meta, reference, block_id, k, b):
    """Permutation b: returns (x', meta') with rows re-sorted by the new section index."""
    s = meta.sample_index.to_numpy(int); NS = int(s.max()) + 1
    ref = np.asarray(reference['labels']).astype(np.int64)
    inv, usec, ugrp = units(s, ref, np.asarray(block_id).astype(np.int64), k)
    rng = np.random.default_rng(np.random.SeedSequence([SEED_BASE, b]).generate_state(1)[0])
    dest = usec.copy()
    for g in np.unique(ugrp):
        ix = np.flatnonzero(ugrp == g); dest[ix] = rng.permutation(usec[ix])
    s1 = dest[inv]
    o_ref = reference['offsets']
    x1 = np.asarray(x) - o_ref[s] + o_ref[s1]
    order = np.argsort(s1, kind='stable')
    names = meta.drop_duplicates('sample_index').sort_values('sample_index').sample_id.to_numpy(str)
    m1 = meta.iloc[order].copy()
    m1['sample_index'] = s1[order]; m1['sample_id'] = names[s1[order]]; m1['global_index'] = np.arange(len(m1))
    return np.ascontiguousarray(x1[order]), m1.reset_index(drop=True)
