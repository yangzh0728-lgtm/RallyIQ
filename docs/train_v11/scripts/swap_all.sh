#!/bin/bash
# After each round's archive copy is written (archive.log), swap its verified subtrees (swap_to_t7.py verifies sha1 itself).
cd ~/Desktop/RallyIQ
for n in 4 5 6 7 8 9 10; do
  until grep -q "^train_v$n bytes_src" train_v11/logs/archive.log; do sleep 60; done
  nice ~/RallyIQ/.venv/bin/python train_v11/scripts/swap_to_t7.py $n --do
done
echo SWAP_ALL_DONE
