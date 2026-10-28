#!/bin/bash
# Pooled DiSCoTME training as run for the paper: 4 nodes x 4 V100 (torchrun), then export of U on the same allocation.
# usage: sbatch encoder/train_slurm.sh CONFIG RUN_DIR U_OUT_DIR       (adjust partition / resources to your cluster)
#SBATCH --job-name=pooled_discotme
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=320G
#SBATCH --time=24:00:00
set -euo pipefail
CONFIG=$1; RUN=$2; OUT=$3; PY=${PYTHON:-python}
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1); MASTER_PORT=29500
launch() {
  srun --ntasks=4 --ntasks-per-node=1 --kill-on-bad-exit=1 $PY -m torch.distributed.run --nnodes=4 --nproc-per-node=4 \
    --rdzv-id=$SLURM_JOB_ID-$1 --rdzv-backend=c10d --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT "${@:2}"
}
RESUME=()
if [ -f "$RUN/checkpoints/latest.json" ]; then
  RESUME=(--resume "$RUN/checkpoints/$($PY -c "import json,sys;print(json.load(open(sys.argv[1]))['checkpoint'])" "$RUN/checkpoints/latest.json")")
fi
launch train encoder/train.py --config "$CONFIG" --run-dir "$RUN" "${RESUME[@]}"
launch extract encoder/extract.py --run-dir "$RUN" --out "$OUT"
