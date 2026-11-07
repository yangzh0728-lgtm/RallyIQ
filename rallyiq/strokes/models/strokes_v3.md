# Stroke-type model (sklearn_hgb, features: v3, source: shuttleset)

rules_v1 on the same shots: accuracy 0.425

Shots: 2486 from 34 groups, 5-fold grouped CV (out-of-fold)

- accuracy: 0.717
- macro F1: 0.586

| type | n | precision | recall | F1 |
| --- | --- | --- | --- | --- |
| short_serve | 112 | 0.83 | 0.86 | 0.84 |
| long_serve | 7 | 0.00 | 0.00 | 0.00 |
| clear | 187 | 0.90 | 0.81 | 0.85 |
| smash | 270 | 0.74 | 0.82 | 0.78 |
| drop | 200 | 0.70 | 0.66 | 0.68 |
| net_shot | 498 | 0.76 | 0.79 | 0.77 |
| cross_net | 121 | 0.54 | 0.55 | 0.54 |
| push | 198 | 0.43 | 0.31 | 0.36 |
| net_kill | 36 | 0.55 | 0.31 | 0.39 |
| drive | 83 | 0.31 | 0.23 | 0.26 |
| lift | 494 | 0.76 | 0.84 | 0.80 |
| block | 280 | 0.74 | 0.76 | 0.75 |

Confusion matrix (rows = truth, columns = predicted):

| | short_serve | long_serve | clear | smash | drop | net_shot | cross_net | push | net_kill | drive | lift | block |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| short_serve | 96 | 4 | 0 | 1 | 0 | 9 | 0 | 2 | 0 | 0 | 0 | 0 |
| long_serve | 2 | 0 | 0 | 0 | 0 | 2 | 0 | 1 | 0 | 0 | 2 | 0 |
| clear | 0 | 0 | 151 | 13 | 8 | 2 | 0 | 0 | 0 | 0 | 13 | 0 |
| smash | 1 | 2 | 7 | 221 | 22 | 3 | 2 | 0 | 0 | 9 | 0 | 3 |
| drop | 2 | 0 | 6 | 32 | 132 | 6 | 7 | 0 | 0 | 9 | 2 | 4 |
| net_shot | 8 | 4 | 0 | 4 | 11 | 395 | 30 | 11 | 1 | 1 | 6 | 27 |
| cross_net | 0 | 0 | 0 | 4 | 3 | 30 | 66 | 7 | 0 | 0 | 3 | 8 |
| push | 0 | 0 | 0 | 2 | 0 | 14 | 2 | 61 | 3 | 12 | 100 | 4 |
| net_kill | 2 | 0 | 2 | 1 | 0 | 11 | 5 | 3 | 11 | 1 | 0 | 0 |
| drive | 1 | 0 | 1 | 15 | 4 | 5 | 1 | 7 | 5 | 19 | 3 | 22 |
| lift | 3 | 0 | 1 | 0 | 1 | 14 | 1 | 47 | 0 | 2 | 417 | 8 |
| block | 1 | 0 | 0 | 5 | 7 | 31 | 9 | 4 | 0 | 8 | 2 | 213 |
