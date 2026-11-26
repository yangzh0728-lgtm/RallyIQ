#!/bin/bash
cd ~/Desktop/RallyIQ/train_v5
for k in 0906_0618 0906_0619 0906_0621 0324_0241 0320_0239 0310_01 0906_0615 0906_0622 0906_0620; do
  [ -f dumps/own_2026$k.npz ] || nice -n 10 ~/RallyIQ/.venv/bin/python scripts/dump.py own_2026$k >> logs/dump.log 2>&1
done
echo ALLDONE >> logs/dump.log
