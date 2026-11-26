#!/bin/bash
cd ~/Desktop/RallyIQ/train_v5; export OMP_NUM_THREADS=1 TRAIN_INPUTS=$PWD/inputs_C; P=~/RallyIQ/.venv/bin/python; T=repo/scripts/train_hits_v3.py
nice $P $T collect --out ckpt/cands6 --root inputs_C --labels labels/amateur_v4_newids.json --ss-train inputs/pub_v5/shuttleset_train_v3 --ss-test inputs/pub_v5/shuttleset_test_v3 --min-shuttle-conf 0.15 0.3 > logs/collect6.log 2>&1 || { echo COLLECT_FAIL; exit 1; }
TH='{"min_candidate_conf":0.1,"keep_threshold":0.3,"ok_threshold":0.7}'
nice $P $T lovo --cache ckpt/cands6 --features v2+all --am-weight 3 --seeds 5 --thresholds "$TH" --labels labels/amateur_v4_newids.json --tmp ckpt/lovo_E5 > logs/lovo_E5.log 2>&1 || { echo LOVO_FAIL; exit 1; }
nice $P $T fit --cache ckpt/cands6 --features v2+all --am-weight 3 --seeds 5 --thresholds "$TH" --out ckpt/lovo_E5/full.json > logs/fit_E5.log 2>&1 || { echo FIT_FAIL; exit 1; }
echo RETRAIN_DONE
