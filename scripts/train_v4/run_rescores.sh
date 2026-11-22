#!/bin/bash
# usage: run_rescores.sh  "name|args" ...   (runs in parallel, nice, 1 thread each)
cd ~/Desktop/RallyIQ/train_v4
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
for spec in "$@"; do
  name="${spec%%|*}"; args="${spec#*|}"
  nice -n 10 ~/RallyIQ/.venv/bin/python scripts/rescore.py "$name" --labels labels/amateur_v3.json $args > "logs/rs_$name.out" 2>&1 &
done
wait
