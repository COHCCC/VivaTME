"""Compartment-conditional adversarial training of the residual adapter (Methods 3.4).

Per epoch: refit centres / offsets / assignments on the current adapted representation (one round of the
updates of Methods 3.3, compartments weighted equally in the offset step), then mini-batches of 1,024 spots with
L = L_keep + 0.1 E||g(u)||^2 + 0.1 L_proto + lambda(t) L_adv. The final representation is the EMA of the adapter.
"""
import copy
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from .compartments import geometry, squared_distances, update_centers, expectation, section_mass, sharing, class_equal_offsets
from .model import Representation, ConditionalAdversary, encode_all, main_loss, proto_loss, CONFIDENCE
from .utils import rng_seed

EPOCHS = 100
ADV_WARM = 10            # epochs before the adversary is switched on
LAMBDA = 10.             # final adversarial weight
EMA_DECAY = .999
ALPHA_PROTO = .1
BATCH = 1024
LR_ADAPTER, LR_DISC, WEIGHT_DECAY = 1e-3, 3e-3, .01


class InvalidRun(RuntimeError):
    pass


def latent_offsets(e, s, p, mu, n_sections):
    w = sharing(section_mass(p, s, n_sections))
    return class_equal_offsets(e, s, p, mu, w)


def make_context(net, X, s, ref_labels, oldq, eta, mu, n_sections):
    """EM refit on the current representation: centres, class-equal offsets, c0, T and soft assignments."""
    feat, _ = encode_all(net, X); f = feat.numpy().astype(float); K = net.k
    if oldq is None:  # first epoch: start from the reference compartments
        p = np.eye(K)[np.maximum(ref_labels, 0)] * (ref_labels >= 0)[:, None]; mu = np.zeros((K, f.shape[1]))
        for k in range(K):
            means = [f[(s == j) & (ref_labels == k)].mean(0) for j in range(n_sections) if np.any((s == j) & (ref_labels == k))]
            mu[k] = np.mean(means, 0) if means else f.mean(0)
        eta, mu = latent_offsets(f, s, p, mu, n_sections)
    else:
        p = oldq[:, :K]
    weights, _, _ = geometry(f, s)
    mu = update_centers(f - eta[s], p, weights, mu); eta, mu = latent_offsets(f, s, p, mu, n_sections)
    nearest = squared_distances(f - eta[s], mu).min(1); c0 = float(np.quantile(nearest, .95)); temp = float(.1 * np.median(nearest))
    if not np.isfinite(temp) or temp <= 0:
        raise InvalidRun('nonpositive_EM_temperature')
    q = expectation(f, s, mu, eta, c0, temp)[0]; labels = q.argmax(1); valid = (labels < K) & (q.max(1) >= CONFIDENCE)
    ctx = dict(s=s, labels=labels, valid=valid, eta_tensor=torch.as_tensor(eta, dtype=torch.float32),
               mu_tensor=torch.as_tensor(mu, dtype=torch.float32), c0=c0, temperature=temp)
    return ctx, q, eta, mu


def train(x, s, ref_labels, knn, has_mnn, k, seed, out, lam=LAMBDA, epochs=EPOCHS, adv_warm=ADV_WARM, threads=8, log=print):
    """Train VivaTME's adapter. x: (n, 512) U; s: section index; ref_labels: initial compartment labels (-1 dustbin);
    knn: (n, 15) within-section Euclidean neighbours of u; has_mnn: spot has a cross-section mutual nearest neighbour.
    Returns (EMA features, training offsets eta). Checkpoints every 5 epochs in `out` and resumes from them."""
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    if (out / 'TRAIN_DONE').exists():
        return np.load(out / 'features.npy'), np.load(out / 'train_em_offsets.npy')
    torch.set_num_threads(threads); torch.manual_seed(seed); np.random.seed(seed)
    s = np.asarray(s, int); NS = int(s.max()) + 1
    X = torch.tensor(np.asarray(x), dtype=torch.float32); n = len(x)
    net = Representation(k); ema = copy.deepcopy(net)
    with torch.random.fork_rng():
        torch.manual_seed(rng_seed(seed, 201)); adv = ConditionalAdversary(k, NS)
    opt = torch.optim.AdamW(net.parameters(), lr=LR_ADAPTER, weight_decay=WEIGHT_DECAY)
    dopt = torch.optim.AdamW(adv.parameters(), lr=LR_DISC, weight_decay=WEIGHT_DECAY)
    pools = [np.flatnonzero(s == j) for j in range(NS)]
    rng = {name: np.random.default_rng(rng_seed(seed, j)) for j, name in enumerate(('shuffle', 'triplet', 'far', 'adversary', 'scale'), 300)}
    per = math.ceil(n / BATCH); cp = out / 'checkpoint.pt'
    eta = np.zeros((NS, 512)); mu = np.zeros((k, 512)); q = None; history = []; begin = 0; scale_const = None
    if cp.exists():
        z = torch.load(cp, map_location='cpu', weights_only=False)
        net.load_state_dict(z['net']); ema.load_state_dict(z['ema']); adv.load_state_dict(z['align']); opt.load_state_dict(z['opt']); dopt.load_state_dict(z['dopt'])
        torch.set_rng_state(z['torch_rng'])
        for key, v in z['rng'].items():
            rng[key].bit_generator.state = v
        eta, mu, q = z['eta'], z['mu'], z['q']; history = z['history']; begin = z['epoch']; scale_const = z['scale_const']
    init_sd = np.asarray(x).std(0); start = time.monotonic()
    for epoch in range(begin, epochs):
        ctx, q, eta, mu = make_context(net, X, s, ref_labels, q, eta, mu, NS)
        if ctx['valid'].mean() < .5:
            raise InvalidRun('valid_pseudo_fraction_below_50_percent')
        hp, participants = adv.pools(ctx, has_mnn)
        if epoch >= adv_warm and (not hp or not participants.any()):
            raise InvalidRun('no_alignment_participants_or_heads')
        if epoch >= adv_warm and scale_const is None:
            # Rescale L_adv once by the ratio of the two losses at activation.
            ix = np.arange(min(BATCH, n))
            with torch.no_grad():
                main, _, _ = main_loss(net, X, ix, s, knn, pools, rng['scale'], NS)
                al, _ = adv.loss(net, X, adv.sample(hp, rng['scale']), ctx, scale=0., mode='critic')
            scale_const = float(main) / float(al.detach())
        order = rng['shuffle'].permutation(n); acc = []
        for bi, start_ix in enumerate(range(0, n, BATCH)):
            ix = order[start_ix:start_ix + BATCH]
            main, regular, f = main_loss(net, X, ix, s, knn, pools, rng['far'], NS)
            pro = proto_loss(f, ctx['mu_tensor'], ctx['eta_tensor'], torch.as_tensor(s[ix]), ctx['c0'], ctx['temperature'])
            strength = 0.; al = f.sum() * 0
            if epoch >= adv_warm:
                progress = ((epoch - adv_warm) * per + bi) / max(1, (epochs - adv_warm) * per)
                strength = lam * (2 / (1 + np.exp(-10 * progress)) - 1)
            loss = main + regular + ALPHA_PROTO * pro
            if hp:
                al, _ = adv.loss(net, X, adv.sample(hp, rng['adversary']), ctx, scale=strength * (scale_const or 0.), mode='grl')
                loss = loss + al
            if not torch.isfinite(loss):
                raise InvalidRun('nonfinite_total_loss')
            opt.zero_grad(set_to_none=True); dopt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if hp:
                dopt.step()
            with torch.no_grad():
                for ep, param in zip(ema.parameters(), net.parameters()):
                    ep.mul_(EMA_DECAY).add_(param, alpha=1 - EMA_DECAY)
            acc.append((len(ix), float(main.detach()), float(pro.detach()), float(al.detach()), strength))
        array = encode_all(net, X)[0].numpy()
        ratio = float((array.std(0) / np.maximum(init_sd, 1e-12)).min())
        if not np.isfinite(array).all() or ratio < .1:
            raise InvalidRun('nonfinite_or_collapsed_representation')
        w = np.array([a[0] for a in acc], float)
        row = dict(epoch=epoch + 1, valid_fraction=float(ctx['valid'].mean()), n_participants=int(participants.sum()), n_heads=len(hp),
                   L_keep=float(np.average([a[1] for a in acc], weights=w)), L_proto=float(np.average([a[2] for a in acc], weights=w)),
                   L_adv=float(np.average([a[3] for a in acc], weights=w)), lambda_t=float(np.average([a[4] for a in acc], weights=w)),
                   min_sd_ratio=ratio, seconds=time.monotonic() - start)
        history.append(row)
        if (epoch + 1) % 5 == 0 or epoch + 1 == epochs:
            z = dict(net=net.state_dict(), ema=ema.state_dict(), align=adv.state_dict(), opt=opt.state_dict(), dopt=dopt.state_dict(),
                     torch_rng=torch.get_rng_state(), rng={key: v.bit_generator.state for key, v in rng.items()}, epoch=epoch + 1,
                     eta=eta, mu=mu, q=q, history=history, scale_const=scale_const)
            tmp = out / 'checkpoint.partial.pt'; torch.save(z, tmp); tmp.replace(cp)
            log(f'seed {seed} epoch {epoch + 1} valid {row["valid_fraction"]:.3f} L_keep {row["L_keep"]:.4g} L_adv {row["L_adv"]:.4g}')
    features = encode_all(ema, X)[0].numpy()
    if not np.isfinite(features).all() or (features.std(0) / np.maximum(init_sd, 1e-12)).min() < .1:
        raise InvalidRun('nonfinite_or_collapsed_EMA')
    np.save(out / 'features.npy', features); np.save(out / 'train_em_offsets.npy', eta)
    torch.save(dict(model=ema.state_dict(), seed=seed), out / 'model_ema.pt')
    (out / 'train_log.json').write_text(json.dumps(history, indent=1))
    (out / 'TRAIN_DONE').touch()
    return features, eta
