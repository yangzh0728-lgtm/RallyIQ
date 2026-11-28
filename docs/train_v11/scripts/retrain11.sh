#!/bin/bash
# H (PLAN.md): v3b recipe (as train_v8 retrain8.sh) on old + new own labels -> ckpt/lovo_$1. usage: retrain11.sh NAME LABELS
cd ~/Desktop/RallyIQ/train_v11; export OMP_NUM_THREADS=1 TRAIN_INPUTS=$HOME/Desktop/RallyIQ/train_v11/inputs PYTHONPATH=$HOME/Desktop/RallyIQ/train_v11/repo; P=~/RallyIQ/.venv/bin/python; T=repo/scripts/train_hits_v3.py; N=$1; LAB=$2
nice $P $T collect --out ckpt/cands_$N --root $TRAIN_INPUTS --labels $LAB --ss-train ${SS_TRAIN:-$HOME/Desktop/RallyIQ/train_v5/inputs/pub_v5/shuttleset_train_v3} --ss-test ~/Desktop/RallyIQ/train_v5/inputs/pub_v5/shuttleset_test_v3 --min-shuttle-conf 0.15 0.3 > logs/collect_$N.log 2>&1 || { echo COLLECT_FAIL; exit 1; }
TH='{"min_candidate_conf":0.1,"keep_threshold":0.3,"ok_threshold":0.7}'
nice $P $T lovo --cache ckpt/cands_$N --features v2+all --am-weight 3 --seeds 5 --thresholds "$TH" --labels $LAB --tmp ckpt/lovo_$N > logs/lovo_$N.log 2>&1 || { echo LOVO_FAIL; exit 1; }
nice $P $T fit --cache ckpt/cands_$N --features v2+all --am-weight 3 --seeds 5 --thresholds "$TH" --out ckpt/lovo_$N/full.json > logs/fit_$N.log 2>&1 || { echo FIT_FAIL; exit 1; }
echo RETRAIN_DONE
