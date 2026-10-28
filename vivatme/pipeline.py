"""End-to-end VivaTME run on one cohort.

  prepare(cohort_dir)          reference fit + spatial blocks + neighbour graphs (shared by all seeds)
  train_seed(cohort_dir, seed) adapter training + compartment read-out
  deviations(cohort_dir, seed, tau)  fixed-assignment deviations at the calibrated threshold
  null_permutation(...)        one permutation of the full-refit null (max z with tau = inf)
"""
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from .utils import load_cohort, save_npz, load_npz, write_json, read_json
from .reference import fit_reference
from .graphs import within_section_knn, mnn_graph
from .train import train, InvalidRun
from .readout import readout
from .null import permuted_cohort

REFERENCE_SEED = 42


def match_to_reference(labels, ref_labels, k):
    """Hungarian maximum-overlap map of read-out compartments onto the reference compartments (C0..C{K-1})."""
    counts = pd.crosstab(labels, ref_labels).reindex(index=range(k), columns=range(k), fill_value=0).to_numpy()
    rr, cc = linear_sum_assignment(-counts)
    return {int(a): int(b) for a, b in zip(rr, cc)}


def prepare_arrays(x, meta, k, n_perm=500, log=print):
    s = meta.sample_index.to_numpy(int)
    ref, warm, warm_stats, tau, block_id = fit_reference(x, meta, k, REFERENCE_SEED, n_perm=n_perm, log=log)
    knn = within_section_knn(x, s)
    mnn = mnn_graph(x, s, ref['labels'] >= 0)
    has_mnn = np.diff(mnn.indptr) > 0
    return dict(reference=ref, warm=warm, warm_stats=warm_stats, tau=tau, block_id=block_id, knn=knn, has_mnn=has_mnn)


def prepare(cohort_dir, U, spots, k, n_perm=500, log=print):
    out = Path(cohort_dir); out.mkdir(parents=True, exist_ok=True)
    if (out / 'reference/DONE').exists():
        return
    t0 = time.monotonic()
    meta, x = load_cohort(U, spots)
    a = prepare_arrays(x, meta, k, n_perm, log)
    save_npz(out / 'reference/fit.npz', **a['reference'])
    save_npz(out / 'reference/compartments.npz', **a['warm'])
    save_npz(out / 'reference/graphs.npz', knn=a['knn'], has_mnn=a['has_mnn'], block_id=a['block_id'])
    write_json(out / 'reference/stats.json', dict(compartments=a['warm_stats'], reference_tau=a['tau'], k=k, U=str(U), spots=str(spots),
                                                    n_spots=len(x), n_sections=int(meta.sample_index.max()) + 1, seconds=time.monotonic() - t0))
    (out / 'reference/DONE').touch()


def _seed_arrays(x, meta, k, seed, a, out, tau_pilot, tau, epochs, adv_warm, log):
    s = meta.sample_index.to_numpy(int); ref = a['reference']
    features, eta = train(x, s, ref['labels'], a['knn'], a['has_mnn'], k, seed, out, epochs=epochs, adv_warm=adv_warm, log=log)
    return readout(x, s, features, ref, k, tau_pilot, tau)


def train_seed(cohort_dir, seed, epochs=100, adv_warm=10, log=print):
    """Adapter training and read-out of one seed; deviations at the reference threshold are a placeholder until the
    full-refit threshold is applied with deviations()."""
    out = Path(cohort_dir); st = read_json(out / 'reference/stats.json'); k = st['k']
    meta, x = load_cohort(st['U'], st['spots'])
    a = dict(reference=load_npz(out / 'reference/fit.npz'), **load_npz(out / 'reference/graphs.npz'))
    d = out / f'seed{seed}'
    if (d / 'compartments.npz').exists():
        return
    tau0 = st['reference_tau']['95']
    fit, b, stats = _seed_arrays(x, meta, k, seed, a, d / 'train', tau0, tau0, epochs, adv_warm, log)
    save_npz(d / 'compartments.npz', **b)
    mapping = match_to_reference(b['labels'], a['reference']['labels'], k)
    write_json(d / 'readout_stats.json', dict(stats, label_to_reference=mapping))
    meta.assign(label=b['labels'], compartment=[f'C{mapping[v]}' if v >= 0 else 'dustbin' for v in b['labels']])[
        ['spot_id', 'sample_id', 'label', 'compartment']].to_csv(d / 'labels.csv', index=False)


def deviations(cohort_dir, seed, tau):
    """Fixed-assignment deviations of one seed at threshold tau (pilot from the reference fit, final from the pilot)."""
    from threadpoolctl import threadpool_limits
    from .readout import deviation_readout
    out = Path(cohort_dir); st = read_json(out / 'reference/stats.json'); k = st['k']
    meta, x = load_cohort(st['U'], st['spots']); s = meta.sample_index.to_numpy(int)
    ref = load_npz(out / 'reference/fit.npz'); b = load_npz(out / f'seed{seed}/compartments.npz')
    with threadpool_limits(8):
        final, pilot, stats = deviation_readout(x, s, b, ref, tau, tau, k)
    final.update(labels=b['labels'], tau=np.array(tau))
    save_npz(out / f'seed{seed}/deviations.npz', **final)
    # One row per (section, compartment): soft mass, standardized deviation z, deviation magnitude, reported (z > tau).
    mapping = {int(a): b_ for a, b_ in read_json(out / f'seed{seed}/readout_stats.json')['label_to_reference'].items()}
    names = meta.drop_duplicates('sample_index').sort_values('sample_index').sample_id.tolist()
    norm = np.linalg.norm(final['delta'], axis=2)
    rows = [dict(sample_id=names[j], compartment=f'C{mapping[kk]}', soft_mass=float(final['section_mass'][j, kk]),
                 z=float(final['z'][kk, j]), deviation_norm=float(norm[kk, j]), eligible=bool(final['eligible'][kk, j]),
                 reported=bool(norm[kk, j] > 0)) for kk in range(k) for j in range(len(names))]
    pd.DataFrame(rows).sort_values('z', ascending=False).to_csv(out / f'seed{seed}/deviations.csv', index=False)
    return final


def null_permutation(cohort_dir, b, epochs=100, adv_warm=10, n_perm_inner=500, log=print):
    """One permutation of the full-refit null; returns the maximum z over eligible cells (tau = inf)."""
    out = Path(cohort_dir); st = read_json(out / 'reference/stats.json'); k = st['k']
    dest = out / 'null' / f'perm_{b:03d}.json'
    if dest.exists():
        return read_json(dest)
    meta, x = load_cohort(st['U'], st['spots'])
    ref = load_npz(out / 'reference/fit.npz'); block_id = load_npz(out / 'reference/graphs.npz')['block_id']
    x1, m1 = permuted_cohort(x, meta, ref, block_id, k, b)
    t0 = time.monotonic(); res = dict(perm=int(b))
    try:
        a = prepare_arrays(x1, m1, k, n_perm_inner, log)
        work = out / 'null_work' / f'perm_{b:03d}'
        fit, _, _ = _seed_arrays(x1, m1, k, REFERENCE_SEED, a, work, float('inf'), float('inf'), epochs, adv_warm, log)
        z = np.asarray(fit['z']); e = np.asarray(fit['eligible'])
        res.update(status='ok', max_z=float(z[e].max()), n_eligible=int(e.sum()))
        import shutil
        shutil.rmtree(work, ignore_errors=True)
    except InvalidRun as exc:
        res.update(status='invalid', reason=str(exc))
    res['seconds'] = time.monotonic() - t0
    write_json(dest, res)
    return res


def summarize_null(cohort_dir, n=500, min_ok=450):
    out = Path(cohort_dir)
    rows = [read_json(p) for p in sorted((out / 'null').glob('perm_*.json'))]
    mx = np.array([r['max_z'] for r in rows if r.get('status') == 'ok'], float)
    if len(mx) < min_ok:
        raise SystemExit(f'only {len(mx)} valid permutations (< {min_ok})')
    tau = {str(q): float(np.quantile(mx, q / 100)) for q in (90, 95, 99)}
    write_json(out / 'null/thresholds.json', dict(n_requested=n, n_ok=int(len(mx)), tau=tau))
    return tau
