"""Residual adapter, gradient reversal and compartment-conditional discriminators (Methods 3.4)."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

DIM = 512
LABEL_SMOOTHING = .1
PER_CLASS = 32          # spots drawn per active (compartment, section) pair and step
MIN_SPOTS = 20          # participating spots for a section to enter a compartment's discriminator
MIN_SECTIONS = 3        # sections needed for a compartment's discriminator to be active
CONFIDENCE = .6         # q_ik threshold of a confidently assigned spot


class Reverse(torch.autograd.Function):
    """Gradient reversal layer: identity forward, -scale * gradient backward."""
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.scale * grad, None


class Adapter(nn.Module):
    """z = u + g(u), g a 512-256-512 SELU perceptron whose output layer starts at zero (identity map)."""
    def __init__(self):
        super().__init__()
        self.g = nn.Sequential(nn.Linear(DIM, 256), nn.SELU(), nn.Linear(256, DIM))
        nn.init.zeros_(self.g[-1].weight); nn.init.zeros_(self.g[-1].bias)

    def forward(self, x):
        return x + self.g(x)


class Representation(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = int(k)
        self.body = Adapter()
        # Unused projection head of the original implementation. It is kept so that parameter initialisation
        # consumes the torch random stream exactly as in the published runs (discriminator dropout draws from it).
        self.project = nn.Linear(DIM, 128)
        self.register_buffer('nu', torch.zeros(self.k, DIM))

    def feature(self, x):
        z = self.body(x)
        return z, z


def encode_all(net, X):
    with torch.no_grad():
        values = [net.feature(X[i:i + 4096]) for i in range(0, len(X), 4096)]
        return tuple(torch.cat([v[j] for v in values]) for j in range(2))


def proto_loss(f, mu, eta, ss, c0, T):
    """L_proto: expected squared distance to the current centres under fixed soft assignments."""
    cost = (f[:, None] - eta[ss, None] - mu[None]).square().sum(2)
    q = F.softmax(-torch.cat([cost, torch.full((len(f), 1), float(c0), device=f.device)], 1) / float(T), 1).detach()
    return (q[:, :mu.shape[0]] * cost).sum(1).mean()


def far_samples(anchors, near, s, section_ids, rng, n_sections):
    """15 random spots of the anchor's section that are neither the anchor nor one of its 15 neighbours."""
    far = np.empty_like(near)
    for sj in range(n_sections):
        rows = np.flatnonzero(s[anchors] == sj)
        if not len(rows):
            continue
        z = rng.choice(section_ids[sj], (len(rows), 15))
        bad = (z == anchors[rows, None]) | np.any(z[:, :, None] == near[rows, None, :], axis=2)
        while bad.any():
            z[bad] = rng.choice(section_ids[sj], bad.sum())
            bad = (z == anchors[rows, None]) | np.any(z[:, :, None] == near[rows, None, :], axis=2)
        far[rows] = z
    return far


def main_loss(net, X, ix, s, knn, pools, rng, n_sections):
    """L_keep (equation 4) and the residual penalty 0.1 E||g(u)||^2; also returns z for the batch anchors."""
    near = knn[ix]; far = far_samples(ix, near, s, pools, rng, n_sections)
    joined = np.r_[ix, near.ravel(), far.ravel()]; unique, inverse = np.unique(joined, return_inverse=True)
    out = net.body(X[unique])[inverse]; n = len(ix)
    a = out[:n]; b = out[n:n + n * 15].reshape(n, 15, DIM); c = out[n + n * 15:].reshape(n, 15, DIM)
    with torch.no_grad():
        on = torch.linalg.vector_norm(X[ix, None] - X[near], dim=2)
        of = torch.linalg.vector_norm(X[ix, None] - X[far], dim=2)
    dn = torch.linalg.vector_norm(a[:, None] - b, dim=2); df = torch.linalg.vector_norm(a[:, None] - c, dim=2)
    keep = ((dn - on).square().mean() + .2 * (df - of).square().mean()) / 1.2
    regular = .1 * (a - X[ix]).square().sum(1).mean()
    return keep, regular, a


def discriminator(n_out):
    return nn.Sequential(nn.Linear(DIM, 128), nn.LeakyReLU(.01), nn.Dropout(.2), nn.Linear(128, n_out))


class ConditionalAdversary(nn.Module):
    """One discriminator per compartment predicting the section of z_i - eta_s (equation 5)."""
    def __init__(self, k, n_sections):
        super().__init__()
        self.k = int(k); self.n_sections = int(n_sections)
        self.heads = nn.ModuleDict({str(j): discriminator(self.n_sections) for j in range(self.k)})

    def pools(self, ctx, has_mnn):
        """Participating spots: confidently assigned and with a cross-section mutual nearest neighbour."""
        label = ctx['labels']; s = ctx['s']; valid = ctx['valid'] & has_mnn; heads = {}
        for k in range(self.k):
            classes = [j for j in range(self.n_sections) if np.sum(valid & (label == k) & (s == j)) >= MIN_SPOTS]
            if len(classes) >= MIN_SECTIONS:
                heads[str(k)] = {j: np.flatnonzero(valid & (label == k) & (s == j)) for j in classes}
        return heads, valid

    def sample(self, pools, rng):
        return {head: {cl: rng.choice(ids, PER_CLASS, replace=True) for cl, ids in pp.items()} for head, pp in pools.items()}

    def loss(self, net, X, batch, ctx, scale=0., mode='grl'):
        values = []; n = 0
        for head, pools in batch.items():
            classes = sorted(pools); idx = np.concatenate([pools[j] for j in classes]); n += len(idx)
            f, _ = net.feature(X[idx])
            feat = f - ctx['eta_tensor'][torch.as_tensor(ctx['s'][idx], device=X.device)]
            feat = feat.detach() if mode == 'critic' else Reverse.apply(feat, float(scale))
            y = torch.cat([torch.full((len(pools[j]),), i, dtype=torch.long, device=X.device) for i, j in enumerate(classes)])
            logits = self.heads[head](feat)[:, classes]
            values.append(F.cross_entropy(logits, y, label_smoothing=LABEL_SMOOTHING, reduction='none').mean())
        if not values:
            raise RuntimeError('no usable alignment heads')
        return torch.stack(values).mean(), dict(n_heads=len(values), n_sampled=n)
