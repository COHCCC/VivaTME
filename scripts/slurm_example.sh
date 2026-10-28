#!/bin/bash
# Run VivaTME on one cohort with SLURM: prepare -> 3 seeds -> 500-permutation null -> tau -> deviations.
# usage: bash scripts/slurm_example.sh configs/her2.yaml      (edit the sbatch options for your cluster)
set -euo pipefail
C=$1; PY=${PYTHON:-python}; R=$(cd "$(dirname "$0")/.." && pwd)
OPT="--cpus-per-task=8 --mem=16G --parsable"
run() { echo "cd $R && OMP_NUM_THREADS=8 $PY scripts/run_vivatme.py $*"; }
p=$(sbatch $OPT --time=04:00:00 --job-name=viva_prep --wrap "$(run prepare --config $C)")
t=$(sbatch $OPT --time=12:00:00 --job-name=viva_train --array=42-44 --dependency=afterok:$p --wrap "$(run train --config $C --seed \$SLURM_ARRAY_TASK_ID)")
n=$(sbatch $OPT --time=12:00:00 --job-name=viva_null --array=0-499%60 --dependency=afterok:$p --wrap "$(run null --config $C)")
s=$(sbatch --parsable --time=00:20:00 --job-name=viva_tau --dependency=afterany:$n --wrap "$(run summarize --config $C)")
sbatch $OPT --time=02:00:00 --job-name=viva_dev --array=42-44 --dependency=afterok:$s:$t --wrap "$(run deviations --config $C --seed \$SLURM_ARRAY_TASK_ID)"
