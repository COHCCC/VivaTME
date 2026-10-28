"""Shared compartments with whole-section offsets (Methods 3.3).

Spot representations u_i are explained by K compartment centres mu_k, one offset o_s per section and a dustbin
with a fixed cost c0. Assignments are soft (temperature T); centres are section-balanced weighted means; offsets
are estimated only from compartments that are present in at least three sections (smooth sharing weight).
"""
import time
import numpy as np
from scipy.special import softmax, expit
from sklearn.cluster import kmeans_plusplus

M_MIN = 30.          # soft mass (spots) for a compartment to count as present in a section
S_SHARED = 3         # number of sections a compartment must be present in to inform the offsets
LAMBDA_OFFSET = 10.  # ridge on the section offsets
TOL = 1e-4           # relative change of centres and offsets at convergence
MAX_ITER = 500
N_INIT = 10          # k-means++ restarts of the initial fit
INIT_SEED_BASE = 20260923


def init_seed(seed, index):
    """Seed of restart `index` (restart 0 uses the run seed itself)."""
    return int(seed) if index == 0 else int(np.random.SeedSequence([INIT_SEED_BASE, seed, index]).generate_state(1)[0])


def geometry(x, sample):
    """Per-spot weight 1/n_s, centred section means, spots per section."""
    sample = np.asarray(sample, np.int64)
    ns = np.bincount(sample)
    assert (ns > 0).all()
    means = np.stack([x[sample == s].mean(0) for s in range(len(ns))])
    return 1. / ns[sample], means - means.mean(0), ns


def squared_distances(x, centers):
    return np.maximum(np.einsum('ij,ij->i', x, x)[:, None]
                      + np.einsum('ij,ij->i', centers, centers)[None, :] - 2 * (x @ centers.T), 0.)


def expectation(x, sample, centers, offsets, c0, temperature):
    """Soft assignment to the K compartments and the dustbin (last column), equation (1)."""
    assert temperature > 0 and np.isfinite(temperature) and np.isfinite(c0)
    distance = squared_distances(x - offsets[sample], centers)
    cost = np.column_stack([distance, np.full(len(x), c0)])
    q = softmax(-cost / temperature, axis=1)
    assert np.isfinite(q).all() and np.allclose(q.sum(1), 1., atol=1e-12)
    return q, distance


def update_centers(y, p, weights, previous):
    """Section-balanced weighted means; compartments without mass keep their previous centre."""
    wp = weights[:, None] * p
    mass = wp.sum(0)
    centers = previous.copy()
    nonzero = mass > 0
    centers[nonzero] = (wp[:, nonzero].T @ y) / mass[nonzero, None]
    return centers


def section_mass(p, sample, n_samples):
    return np.stack([p[sample == s].sum(0) for s in range(n_samples)])


def sharing(mass, m_min=M_MIN, s_shared=S_SHARED):
    """Smooth indicator (in [0, 1]) that a compartment is present in at least `s_shared` sections."""
    return expit((expit((mass - m_min) / 5.).sum(0) - s_shared + .5) / .25)


def center_gauge(offsets, centers):
    """Centre the offsets and move the common translation into the centres (distances unchanged)."""
    mean = offsets.mean(0)
    return offsets - mean, centers + mean


def update_offsets(x, sample, p, centers, shared, shrink):
    assert shrink >= 0
    n_samples = int(sample.max()) + 1
    offsets = np.zeros((n_samples, x.shape[1]), np.float64)
    for s in range(n_samples):
        ids = np.flatnonzero(sample == s)
        a = p[ids][:, shared]
        support = float(a.sum())
        if support + shrink > 0:
            offsets[s] = (a.sum(1) @ x[ids] - a.sum(0) @ centers[shared]) / (support + shrink)
    return center_gauge(offsets, centers)


def weighted_offsets(x, sample, p, centers, weights, shrink):
    """Offsets of equation (2) with mass-weighted compartments; returns (offsets, centres)."""
    ids = np.flatnonzero(weights > 0)
    adjusted_p = p.copy()
    adjusted_p[:, ids] *= weights[ids]
    return update_offsets(x, sample, adjusted_p, centers, ids, shrink)


def class_equal_offsets(x, sample, p, centers, weights, shrink=LAMBDA_OFFSET, m_min=M_MIN):
    """Offsets with compartments weighted equally (used during adapter training); returns (offsets, centres)."""
    ns = int(sample.max()) + 1
    raw = np.zeros((ns, x.shape[1]), dtype=np.float64)
    for s in range(ns):
        take = sample == s
        probability = p[take]
        mass = probability.sum(0)
        eligible = (mass >= m_min) & (weights > 0)
        effective = float(weights[eligible] @ mass[eligible])
        if eligible.any():
            residuals = probability[:, eligible].T @ x[take] / mass[eligible, None] - centers[eligible]
            value = weights[eligible] @ residuals / weights[eligible].sum()
        else:
            value = np.zeros(x.shape[1])
        raw[s] = (effective * value) / (effective + shrink)
    return center_gauge(raw, centers)


def fit_compartments(x, sample, k, seed, q_dustbin=.95, temperature_multiplier=.1):
    """One k-means++-initialised fit of the shared-compartment model (one restart)."""
    started = time.monotonic()
    x = np.asarray(x, np.float64); sample = np.asarray(sample, np.int64)
    assert x.ndim == 2 and np.isfinite(x).all()
    weights, translated, ns = geometry(x, sample)
    n_samples = len(ns)
    offsets = translated.copy()
    centers, _ = kmeans_plusplus(x - offsets[sample], n_clusters=k, sample_weight=weights, random_state=int(seed))
    centers = centers.copy()
    nearest = squared_distances(x - offsets[sample], centers).min(1)
    c0 = float(np.quantile(nearest, q_dustbin))
    temperature = float(np.median(nearest) * temperature_multiplier)
    if temperature <= 0 or not np.isfinite(temperature):
        raise ValueError('Nonpositive initialization-based temperature')
    converged = False
    for iteration in range(1, MAX_ITER + 1):
        q, distance = expectation(x, sample, centers, offsets, c0, temperature)
        p = q[:, :k]
        weight = sharing(section_mass(p, sample, n_samples))
        new_centers = update_centers(x - offsets[sample], p, weights, centers)
        new_offsets, new_centers = weighted_offsets(x, sample, p, new_centers, weight, LAMBDA_OFFSET)
        dc = float(np.linalg.norm(new_centers - centers) / max(np.linalg.norm(centers), 1e-12))
        do = float(np.linalg.norm(new_offsets - offsets) / max(np.linalg.norm(offsets), 1e-12))
        centers, offsets = new_centers, new_offsets
        if dc < TOL and do < TOL:
            converged = True; break
    # Compartments without M_MIN soft mass in any section are dropped before the final assignment.
    before_q, _ = expectation(x, sample, centers, offsets, c0, temperature)
    active = np.flatnonzero((section_mass(before_q[:, :k], sample, n_samples) >= M_MIN).any(0))
    q, distance = expectation(x, sample, centers[active], offsets, c0, temperature)
    choice = q.argmax(1)
    labels = np.full(len(x), -1, np.int64)
    accepted = choice < len(active)
    labels[accepted] = active[choice[accepted]]
    mass = np.zeros((n_samples, k))
    mass[:, active] = section_mass(q[:, :-1], sample, n_samples)
    fit = dict(centers_full=centers, centers=centers[active], active_ids=active, offsets=offsets,
               q=q.astype(np.float32), labels=labels, section_mass=mass)
    stats = dict(c0=c0, temperature=temperature, converged=converged, n_iter=iteration, k_retained=len(active),
                 expected_cost_final=float(weights @ ((q[:, :-1] * distance).sum(1) + q[:, -1] * c0) / weights.sum()),
                 seconds=time.monotonic() - started, initialization_seed=int(seed))
    return fit, stats


def fit_compartments_multi(x, sample, k, seed=42, n_init=N_INIT):
    """N_INIT restarts; keeps the restart with the lowest section-balanced expected cost (ties: lowest index)."""
    best = None
    for index in range(n_init):
        fit, stats = fit_compartments(x, sample, k, init_seed(seed, index))
        stats['initialization_index'] = index
        if best is None or stats['expected_cost_final'] < best[1]['expected_cost_final']:
            best = (fit, stats)
    best[1]['selected_initialization'] = best[1]['initialization_index']
    return best


def full_q(fit):
    """(n, K+1) assignment matrix with the pruned compartments as zero columns and the dustbin last."""
    k = fit['centers_full'].shape[0]
    q = np.zeros((len(fit['labels']), k + 1))
    q[:, fit['active_ids']] = fit['q'][:, :-1]
    q[:, -1] = fit['q'][:, -1]
    return q
