"""Compartment deviations (Methods 3.5).

Each compartment mean of each section is decomposed as  mean_{sk} ~ mu_k + o_s + delta_{sk}; delta is a group
soft-threshold of the residual r_{sk} with the test statistic z_{sk} = sqrt(m_{sk}) ||r_{sk}|| (equation 8).
`fit_joint` is the joint fit (assignments updated) used for the initial reference; `fit_fixed` keeps the
assignments fixed (read-out of VivaTME and of every null permutation).
"""
import time
import types
import numpy as np
from scipy.special import softmax
from .compartments import squared_distances, geometry, center_gauge, sharing, M_MIN, S_SHARED, LAMBDA_OFFSET, TOL, MAX_ITER


def expectation(x, sections, centers, offsets, delta, c0, temperature, active=None):
    k = len(centers); distance = np.empty((len(x), k))
    for s in range(len(offsets)):
        ids = np.flatnonzero(sections == s)
        distance[ids] = squared_distances(x[ids] - offsets[s], centers + delta[s])
    cost = np.column_stack([distance, np.full(len(x), c0)])
    if active is not None:
        absent = np.ones(k, bool); absent[active] = False; cost[:, np.flatnonzero(absent)] = np.inf
    q = softmax(-cost / temperature, axis=1)
    assert np.isfinite(q).all()
    return q, distance


def sufficient(x, sections, p, n_sections):
    """Soft mass m_{sk} and soft sums of every (section, compartment) cell."""
    mass = np.zeros((n_sections, p.shape[1]))
    sums = np.zeros((n_sections, p.shape[1], x.shape[1]))
    for s in range(n_sections):
        use = sections == s; mass[s] = p[use].sum(0); sums[s] = p[use].T @ x[use]
    return mass, sums


def delta_step(mass, sums, centers, offsets, tau, m_min=M_MIN):
    """Equation (8): residual r, statistic z, eligibility (m >= m_min) and the soft-thresholded deviation."""
    r = np.divide(sums, mass[:, :, None], out=np.zeros_like(sums), where=mass[:, :, None] > 0)
    r -= centers[None] + offsets[:, None]
    z = np.sqrt(mass) * np.linalg.norm(r, axis=2)
    eligible = mass >= m_min
    factor = np.zeros_like(z)
    use = eligible & (z > tau)
    factor[use] = 1 - tau / z[use]
    return r * factor[:, :, None], r, z, eligible


def m_step(x, sections, p, centers, offsets, delta, tau):
    """Block updates of centres, offsets and deviations with the current deviations subtracted."""
    ns = np.bincount(sections, minlength=len(offsets))
    mass, sums = sufficient(x, sections, p, len(offsets))
    weighted_mass = (mass / ns[:, None]).sum(0)
    numerator = ((sums - mass[:, :, None] * (offsets[:, None] + delta)) / ns[:, None, None]).sum(0)
    mu = centers.copy()
    use = weighted_mass > 0
    mu[use] = numerator[use] / weighted_mass[use, None]
    w = sharing(mass, M_MIN, S_SHARED)
    neff = (mass * w[None]).sum(1)
    resid_sum = sums - mass[:, :, None] * (mu[None] + delta)
    o = (resid_sum * w[None, :, None]).sum(1) / (neff[:, None] + LAMBDA_OFFSET)
    o, mu = center_gauge(o, mu)
    d, r, z, eligible = delta_step(mass, sums, mu, o, tau, M_MIN)
    return mu, o, d, dict(mass=mass, sums=sums, sharing=w, n_effective=neff, residual=r, z=z, eligible=eligible)


def fit_joint(x, sections, warm, warm_stats, tau, max_iter=MAX_ITER):
    """Joint E/M fit of compartments, offsets and deviations, warm-started from a compartment fit."""
    started = time.monotonic()
    x = np.asarray(x, dtype=float); sections = np.asarray(sections, int)
    weights, _, ns = geometry(x, sections)
    mu = warm['centers_full'].copy(); offsets = warm['offsets'].copy()
    delta = np.zeros((len(offsets), len(mu), x.shape[1]), dtype=float)
    c0 = float(warm_stats['c0']); temperature = float(warm_stats['temperature']); tau = float(tau)
    converged = False
    for iteration in range(1, int(max_iter) + 1):
        q, dist = expectation(x, sections, mu, offsets, delta, c0, temperature)
        new_mu, new_o, new_d, info = m_step(x, sections, q[:, :-1], mu, offsets, delta, tau)
        changes = [float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12))
                   for a, b in ((new_mu, mu), (new_o, offsets), (new_d, delta))]
        mu, offsets, delta = new_mu, new_o, new_d
        if max(changes) < TOL:
            converged = True; break
    final_q, dist = expectation(x, sections, mu, offsets, delta, c0, temperature)
    mass_final, _ = sufficient(x, sections, final_q[:, :-1], len(offsets))
    active = np.flatnonzero((mass_final >= M_MIN).any(0))
    if len(active) != len(mu):
        final_q, dist = expectation(x, sections, mu, offsets, delta, c0, temperature, active)
    labels = final_q.argmax(1).astype(int); labels[labels == len(mu)] = -1
    fit = dict(centers_full=mu, centers=mu[active], offsets=offsets, delta=delta.transpose(1, 0, 2), active_ids=active,
               q=final_q[:, np.r_[active, len(mu)]].astype(np.float32), labels=labels, z=info['z'].T,
               eligible=info['eligible'].T, residual=info['residual'].transpose(1, 0, 2))
    stats = dict(converged=converged, n_iter=iteration, seconds=time.monotonic() - started, c0=c0, temperature=temperature,
                 tau=tau, k_retained=len(active), dustbin_fraction=float((labels < 0).mean()))
    return fit, stats


def fit_fixed(x, s, p, mu, o, tau, initial_delta=None, zero_delta=False, n_sections=None):
    """Fixed-assignment decomposition (equation 7): assignments p (n x K, no dustbin column) are held fixed.

    Returns arrays with axes (compartment, section, ...): delta, z, eligible, residual, section_mass."""
    ns_ = int(n_sections if n_sections is not None else len(o))
    p = np.asarray(p, float); x = np.asarray(x, float)
    mass, sums = sufficient(x, s, p, ns_)
    # The sufficient statistics do not depend on (mu, o, delta) once p is fixed: compute them once.
    cached = types.FunctionType(m_step.__code__, dict(m_step.__globals__, sufficient=lambda *args: (mass, sums)), m_step.__name__)
    d = np.zeros((ns_, p.shape[1], x.shape[1])) if initial_delta is None else np.asarray(initial_delta).transpose(1, 0, 2).copy()
    mu = mu.copy(); o = o.copy()
    for it in range(1, MAX_ITER + 1):
        nm, no, nd, info = cached(x, s, p, mu, o, d, float('inf') if zero_delta else tau)
        change = [float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12)) for a, b in ((nm, mu), (no, o), (nd, d))]
        mu, o, d = nm, no, nd
        if not all(np.isfinite(a).all() for a in (mu, o, d)):
            raise FloatingPointError('Nonfinite fixed-assignment M step')
        if max(change) < TOL:
            break
    fit = dict(centers_full=mu, offsets=o, delta=d.transpose(1, 0, 2), section_mass=info['mass'], z=info['z'].T,
               eligible=info['eligible'].T, residual=info['residual'].transpose(1, 0, 2))
    return fit, dict(n_iter=it, converged=max(change) < TOL, changes=change)
