"""VivaTME command-line driver.

  python scripts/run_vivatme.py prepare   --config configs/her2.yaml
  python scripts/run_vivatme.py train     --config configs/her2.yaml --seed 42      (also 43, 44)
  python scripts/run_vivatme.py null      --config configs/her2.yaml --index 0      (0..499; one SLURM array task each)
  python scripts/run_vivatme.py summarize --config configs/her2.yaml
  python scripts/run_vivatme.py deviations --config configs/her2.yaml --seed 42     (uses null/thresholds.json tau95)

Outputs go to <out>/<cohort>/ (see README.md, 'Outputs').
"""
import argparse
import os
import sys
from pathlib import Path
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vivatme import pipeline  # noqa: E402
from vivatme.utils import read_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['prepare', 'train', 'null', 'summarize', 'deviations'])
    ap.add_argument('--config', required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--index', type=int, default=None, help='null permutation (default: SLURM_ARRAY_TASK_ID)')
    ap.add_argument('--tau', type=float, default=None, help='deviation threshold (default: tau95 of the full-refit null)')
    ap.add_argument('--smoke', action='store_true', help='short schedule for a code-path test (15 epochs, 2 permutations)')
    a = ap.parse_args()
    c = yaml.safe_load(Path(a.config).read_text())
    root = Path(os.path.expandvars(c.get('data_root', '.')))
    out = Path(os.path.expandvars(c.get('out', 'results'))) / c['cohort']
    sched = dict(epochs=15, adv_warm=3) if a.smoke else dict(epochs=100, adv_warm=10)
    n_perm = 2 if a.smoke else 500
    if a.stage == 'prepare':
        pipeline.prepare(out, root / c['U'], root / c['spots'], int(c['K']), n_perm=n_perm)
    elif a.stage == 'train':
        pipeline.train_seed(out, a.seed, **sched)
    elif a.stage == 'null':
        b = a.index if a.index is not None else int(os.environ['SLURM_ARRAY_TASK_ID'])
        print(pipeline.null_permutation(out, b, n_perm_inner=n_perm, **sched), flush=True)
    elif a.stage == 'summarize':
        n = 1 if a.smoke else 500
        print(pipeline.summarize_null(out, n=n, min_ok=1 if a.smoke else 450), flush=True)
    else:
        tau = a.tau if a.tau is not None else read_json(out / 'null/thresholds.json')['tau']['95']
        f = pipeline.deviations(out, a.seed, tau)
        print('tau', tau, 'reported cells', int((f['delta'] ** 2).sum(-1).astype(bool).sum()), flush=True)


if __name__ == '__main__':
    main()
