#!/bin/bash
# retrain the hit scorer (v3b recipe) on INPUTS with amateur_v6 labels -> ckpt/lovo_$1
cd ~/Desktop/RallyIQ/train_v8; export ONSET_DIR=$HOME/Desktop/RallyIQ/train_v8/ckpt/onset OMP_NUM_THREADS=1 TRAIN_INPUTS=$2; P=~/RallyIQ/.venv/bin/python; T=repo/scripts/train_hits_v3.py; N=$1
nice $P $T collect --out ckpt/cands_$N --root $2 --labels labels/amateur_v6_newids.json --ss-train ~/Desktop/RallyIQ/train_v5/inputs/pub_v5/shuttleset_train_v3 --ss-test ~/Desktop/RallyIQ/train_v5/inputs/pub_v5/shuttleset_test_v3 --min-shuttle-conf 0.15 0.3 > logs/collect_$N.log 2>&1 || { echo COLLECT_FAIL; exit 1; }
TH='{"min_candidate_conf":0.1,"keep_threshold":0.3,"ok_threshold":0.7}'
nice $P $T lovo --cache ckpt/cands_$N --features $3 --am-weight 3 --seeds 5 --thresholds "$TH" --labels labels/amateur_v6_newids.json --tmp ckpt/lovo_$N > logs/lovo_$N.log 2>&1 || { echo LOVO_FAIL; exit 1; }
nice $P $T fit --cache ckpt/cands_$N --features $3 --am-weight 3 --seeds 5 --thresholds "$TH" --out ckpt/lovo_$N/full.json > logs/fit_$N.log 2>&1 || { echo FIT_FAIL; exit 1; }
echo RETRAIN_DONE
