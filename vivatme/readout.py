"""Read-out (Methods 3.4 'Training and read-out', 3.5).

Compartment labels: refit the shared-compartment model on the EMA features, initialised from the reference labels,
dustbin retained. Deviations: fixed-assignment decomposition on the pre-alignment U with those assignments held
fixed, run twice (from the reference fit, then from its own result).
"""
import time
import numpy as np
from threadpoolctl import threadpool_limits
from .compartments import geometry, section_mass, sharing, weighted_offsets, squared_distances, expectation, update_centers, full_q, \
    M_MIN, LAMBDA_OFFSET, TOL, MAX_ITER
from .deviation import fit_fixed


def compartment_readout(x, s, labels, k):
    """Shared-compartment fit on the aligned features, warm-started from hard labels (class-equal section means)."""
    started = time.monotonic()
    x = np.asarray(x, float); n, d = x.shape; NS = int(s.max()) + 1; weights, _, _ = geometry(x, s)
    p0 = np.eye(k)[np.maximum(labels, 0)] * (labels >= 0)[:, None]; mu = np.zeros((k, d))
    for kk in range(k):
        means = [x[(s == j) & (labels == kk)].mean(0) for j in range(NS) if np.any((s == j) & (labels == kk))]
        mu[kk] = np.mean(means, 0) if means else np.mean([x[s == j].mean(0) for j in range(NS)], 0)
    o, mu = weighted_offsets(x, s, p0, mu, sharing(section_mass(p0, s, NS)), LAMBDA_OFFSET)
    near = squared_distances(x - o[s], mu).min(1); c0 = float(np.quantile(near, .95)); T = float(np.median(near) * .1)
    if not np.isfinite(T) or T <= 0:
        raise FloatingPointError('Nonpositive read-out temperature')
    rel = [np.inf, np.inf]
    for it in range(1, MAX_ITER + 1):
        q, dist = expectation(x, s, mu, o, c0, T); p = q[:, :k]
        nm = update_centers(x - o[s], p, weights, mu); no, nm = weighted_offsets(x, s, p, nm, sharing(section_mass(p, s, NS)), LAMBDA_OFFSET)
        rel = [float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12)) for a, b in [(nm, mu), (no, o)]]
        if not np.isfinite(nm).all() or not np.isfinite(no).all():
            raise FloatingPointError('Nonfinite read-out')
        mu, o = nm, no
        if max(rel) < TOL:
            break
    q, dist = expectation(x, s, mu, o, c0, T); keep = np.flatnonzero((section_mass(q[:, :k], s, NS) >= M_MIN).any(0))
    sub, dist = expectation(x, s, mu[keep], o, c0, T); choice = sub.argmax(1)
    lab = np.full(n, -1, int); use = choice < len(keep); lab[use] = keep[choice[use]]
    mass = np.zeros((NS, k)); mass[:, keep] = section_mass(sub[:, :-1], s, NS)
    fit = dict(centers_full=mu, centers=mu[keep], offsets=o, active_ids=keep, q=sub.astype(np.float32), labels=lab, section_mass=mass)
    return fit, dict(converged=max(rel) < TOL, n_iter=it, c0=c0, temperature=T, seconds=time.monotonic() - started)


def deviation_readout(x, s, readout_fit, reference, tau_pilot, tau, k):
    """Fixed-assignment deviations on U: pilot from the reference fit (at tau_pilot), final from the pilot (at tau)."""
    p = full_q(readout_fit)[:, :k]
    pilot, ps = fit_fixed(x, s, p, reference['centers_full'], reference['offsets'], tau_pilot, initial_delta=reference.get('delta'))
    final, fs = fit_fixed(x, s, p, pilot['centers_full'], pilot['offsets'], tau, initial_delta=pilot['delta'])
    return final, pilot, dict(pilot=ps, final=fs)


def readout(x, s, features, reference, k, tau_pilot, tau, threads=8):
    """Labels from the aligned features, deviations from U. Returns (deviation fit incl. labels, compartment fit)."""
    with threadpool_limits(threads):
        b, bs = compartment_readout(features, s, reference['labels'], k)
        final, pilot, stats = deviation_readout(x, s, b, reference, tau_pilot, tau, k)
    final.update(q=b['q'], active_ids=b['active_ids'], labels=b['labels'], offsets_representation=b['offsets'])
    return final, b, dict(compartments=bs, **stats)
