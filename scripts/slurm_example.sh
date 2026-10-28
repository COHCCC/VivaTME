#!/bin/bash
# Run VivaTME on one cohort with SLURM: run (compartments + deviations) and the 500-permutation null in parallel,
# then the calibrated threshold and the final deviations.
# usage: bash scripts/slurm_example.sh configs/mycohort.yaml      (edit the sbatch options for your cluster)
set -euo pipefail
C=$1; PY=${PYTHON:-python}; R=$(cd "$(dirname "$0")/.." && pwd)
OPT="--cpus-per-task=8 --mem=16G --parsable"
cmd() { echo "cd $R && OMP_NUM_THREADS=8 $PY scripts/run_vivatme.py $*"; }
p=$(sbatch $OPT --time=04:00:00 --job-name=viva_prep --wrap "$(cmd prepare --config $C)")
r=$(sbatch $OPT --time=12:00:00 --job-name=viva_run --dependency=afterok:$p --wrap "$(cmd run --config $C)")
n=$(sbatch $OPT --time=12:00:00 --job-name=viva_null --array=0-499%60 --dependency=afterok:$p --wrap "$(cmd null --config $C)")
s=$(sbatch --parsable --time=00:20:00 --job-name=viva_tau --dependency=afterany:$n --wrap "$(cmd summarize --config $C)")
sbatch $OPT --time=02:00:00 --job-name=viva_dev --dependency=afterok:$s:$r --wrap "$(cmd deviations --config $C)"
