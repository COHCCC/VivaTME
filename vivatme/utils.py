"""Input loading and small helpers."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

SPOT_COLUMNS = ['global_index', 'spot_id', 'sample_id', 'sample_index', 'x_um', 'y_um']


def rng_seed(seed, component):
    """Independent integer seed for one random stream of a run."""
    return int(np.random.SeedSequence([int(seed), int(component)]).generate_state(1)[0])


def load_cohort(U, spots):
    """Read the frozen representation U (n x 512) and the spots table.

    spots.csv needs the columns global_index (0..n-1, row order of U), spot_id (unique), sample_id, sample_index
    (0..S-1, rows sorted by it) and x_um, y_um (micrometres). Annotation columns are never read here."""
    m = pd.read_csv(spots, keep_default_na=False, usecols=lambda c: c in SPOT_COLUMNS, dtype={'sample_id': str, 'spot_id': str})
    for c in ('x_um', 'y_um'):
        m[c] = pd.to_numeric(m[c]).astype(float)
    m = m[SPOT_COLUMNS].reset_index(drop=True)
    x = np.load(U, mmap_mode='r')
    n = len(m)
    assert x.shape == (n, 512), (x.shape, n)
    assert np.array_equal(m.global_index.to_numpy(), np.arange(n))
    s = m.sample_index.to_numpy(int)
    assert (np.diff(s) >= 0).all() and set(np.unique(s)) == set(range(s.max() + 1))
    assert m.groupby('sample_index').sample_id.nunique().eq(1).all() and m.spot_id.is_unique
    assert np.isfinite(m[['x_um', 'y_um']].to_numpy()).all()
    return m, np.asarray(x)


def save_npz(path, **arrays):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.stem + '.partial.npz'); np.savez_compressed(tmp, **arrays); tmp.replace(p)


def load_npz(path):
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def write_json(path, value):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=1, default=lambda v: v.item() if hasattr(v, 'item') else str(v)) + '\n')


def read_json(path):
    return json.loads(Path(path).read_text())
