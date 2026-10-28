"""Neighbour graphs used by the adapter (Methods 3.4).

- within-section Euclidean 15-NN of u (L_keep);
- cross-section mutual nearest neighbours (5 cosine-nearest spots on section-mean-centred u, every pair of
  sections, dustbin spots of the initial fit excluded): a spot takes part in the adversary only if it has one.
"""
import numpy as np
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits
from .compartments import geometry


def within_section_knn(x, s, k=15, threads=8):
    s = np.asarray(s, int); ans = np.empty((len(x), k), int)
    with threadpool_limits(threads):
        for sj in range(int(s.max()) + 1):
            ids = np.flatnonzero(s == sj)
            q = NearestNeighbors(n_neighbors=k + 1, algorithm='brute', n_jobs=6).fit(x[ids]).kneighbors(x[ids], return_distance=False)
            for i, old_i in enumerate(ids):
                ans[old_i] = ids[q[i][q[i] != i][:k]]
    return ans


def _top_indices(sim, k):
    k = min(k, sim.shape[1]); i = np.argpartition(-sim, k - 1, axis=1)[:, :k]; v = np.take_along_axis(sim, i, axis=1)
    order = np.lexsort((i, -v), axis=1)
    return np.take_along_axis(i, order, axis=1)


def _pair_neighbors(x, y, k=10):
    a = np.asarray(x, float); b = np.asarray(y, float)
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12); b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    sim = a @ b.T
    return _top_indices(sim, k), _top_indices(sim.T, k)


def _mutual_edges(ab, ba, k):
    k1 = min(k, ab.shape[1]); k2 = min(k, ba.shape[1]); ii = np.repeat(np.arange(len(ab)), k1); jj = ab[:, :k1].reshape(-1)
    keep = np.any(ba[jj, :k2] == ii[:, None], axis=1)
    return ii[keep], jj[keep]


def mnn_graph(x, s, accepted, k=5, threads=8):
    """Symmetric 0/1 sparse graph of cross-section mutual k-nearest neighbours on section-mean-centred x."""
    s = np.asarray(s, int); NS = int(s.max()) + 1; n = len(x)
    y = x - geometry(x, s)[1][s]
    names = [np.flatnonzero(accepted & (s == j)) for j in range(NS)]; a_all, b_all = [], []
    with threadpool_limits(threads):
        for a in range(NS):
            for b in range(a + 1, NS):
                ids, target = names[a], names[b]
                if not len(ids) or not len(target):
                    continue
                ab, ba = _pair_neighbors(y[ids], y[target], 10)
                left, right = _mutual_edges(ab, ba, k)
                a_all.extend(ids[left].tolist()); b_all.extend(target[right].tolist())
    a_all = np.asarray(a_all, int); b_all = np.asarray(b_all, int)
    z = sparse.csr_matrix((np.ones(2 * len(a_all), np.uint8), (np.r_[a_all, b_all], np.r_[b_all, a_all])), shape=(n, n))
    z.sum_duplicates(); z.data[:] = 1
    return z
