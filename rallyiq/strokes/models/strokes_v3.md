# Stroke-type model (sklearn_hgb, features: v3, source: shuttleset)

rules_v1 on the same shots: accuracy 0.425

Shots: 2486 from 34 groups, 5-fold grouped CV (out-of-fold)

- accuracy: 0.730
- macro F1: 0.591

| type | n | precision | recall | F1 |
| --- | --- | --- | --- | --- |
| short_serve | 112 | 0.82 | 0.85 | 0.83 |
| long_serve | 7 | 0.00 | 0.00 | 0.00 |
| clear | 187 | 0.92 | 0.82 | 0.87 |
| smash | 270 | 0.77 | 0.83 | 0.80 |
| drop | 200 | 0.73 | 0.69 | 0.71 |
| net_shot | 498 | 0.74 | 0.81 | 0.77 |
| cross_net | 121 | 0.57 | 0.45 | 0.50 |
| push | 198 | 0.48 | 0.31 | 0.38 |
| net_kill | 36 | 0.69 | 0.25 | 0.37 |
| drive | 83 | 0.37 | 0.24 | 0.29 |
| lift | 494 | 0.75 | 0.89 | 0.81 |
| block | 280 | 0.73 | 0.79 | 0.76 |

Confusion matrix (rows = truth, columns = predicted):

| | short_serve | long_serve | clear | smash | drop | net_shot | cross_net | push | net_kill | drive | lift | block |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| short_serve | 95 | 0 | 0 | 0 | 0 | 13 | 0 | 2 | 0 | 0 | 2 | 0 |
| long_serve | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 2 | 0 |
| clear | 0 | 0 | 154 | 20 | 2 | 0 | 0 | 1 | 0 | 0 | 10 | 0 |
| smash | 1 | 0 | 4 | 223 | 21 | 5 | 0 | 0 | 0 | 9 | 5 | 2 |
| drop | 2 | 0 | 6 | 25 | 138 | 12 | 6 | 0 | 0 | 5 | 2 | 4 |
| net_shot | 10 | 1 | 0 | 4 | 9 | 401 | 30 | 4 | 0 | 1 | 10 | 28 |
| cross_net | 0 | 0 | 0 | 4 | 3 | 33 | 55 | 6 | 0 | 0 | 2 | 18 |
| push | 0 | 0 | 0 | 0 | 3 | 11 | 3 | 61 | 1 | 8 | 107 | 4 |
| net_kill | 2 | 0 | 0 | 0 | 0 | 6 | 2 | 11 | 9 | 4 | 0 | 2 |
| drive | 1 | 0 | 1 | 9 | 6 | 8 | 0 | 10 | 3 | 20 | 6 | 19 |
| lift | 0 | 0 | 2 | 0 | 0 | 21 | 0 | 26 | 0 | 0 | 439 | 6 |
| block | 1 | 0 | 0 | 5 | 7 | 32 | 1 | 4 | 0 | 7 | 2 | 221 |
