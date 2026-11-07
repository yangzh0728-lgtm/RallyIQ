# Stroke-type model (sklearn_hgb, features: v2, source: shuttleset)

rules_v1 on the same shots: accuracy 0.417

Shots: 1270 from 19 groups, 5-fold grouped CV (out-of-fold)

- accuracy: 0.680
- macro F1: 0.553

| type | n | precision | recall | F1 |
| --- | --- | --- | --- | --- |
| short_serve | 59 | 0.81 | 0.86 | 0.84 |
| long_serve | 3 | 0.00 | 0.00 | 0.00 |
| clear | 95 | 0.84 | 0.74 | 0.79 |
| smash | 136 | 0.80 | 0.82 | 0.81 |
| drop | 78 | 0.62 | 0.63 | 0.62 |
| net_shot | 259 | 0.71 | 0.74 | 0.72 |
| cross_net | 67 | 0.57 | 0.52 | 0.55 |
| push | 104 | 0.43 | 0.31 | 0.36 |
| net_kill | 18 | 0.50 | 0.28 | 0.36 |
| drive | 43 | 0.22 | 0.14 | 0.17 |
| lift | 260 | 0.74 | 0.81 | 0.77 |
| block | 148 | 0.61 | 0.70 | 0.65 |

Confusion matrix (rows = truth, columns = predicted):

| | short_serve | long_serve | clear | smash | drop | net_shot | cross_net | push | net_kill | drive | lift | block |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| short_serve | 51 | 1 | 0 | 0 | 0 | 4 | 1 | 2 | 0 | 0 | 0 | 0 |
| long_serve | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| clear | 0 | 2 | 70 | 8 | 4 | 0 | 0 | 0 | 0 | 0 | 9 | 2 |
| smash | 0 | 1 | 7 | 111 | 8 | 1 | 1 | 0 | 3 | 4 | 0 | 0 |
| drop | 0 | 0 | 6 | 10 | 49 | 5 | 4 | 0 | 0 | 1 | 0 | 3 |
| net_shot | 5 | 1 | 0 | 2 | 8 | 191 | 14 | 3 | 0 | 1 | 14 | 20 |
| cross_net | 0 | 0 | 0 | 2 | 2 | 12 | 35 | 4 | 1 | 0 | 2 | 9 |
| push | 2 | 0 | 0 | 1 | 0 | 12 | 1 | 32 | 0 | 4 | 48 | 4 |
| net_kill | 1 | 0 | 0 | 1 | 0 | 6 | 1 | 0 | 5 | 2 | 0 | 2 |
| drive | 0 | 0 | 0 | 3 | 4 | 2 | 2 | 6 | 0 | 6 | 0 | 20 |
| lift | 0 | 1 | 0 | 1 | 0 | 12 | 0 | 27 | 1 | 2 | 210 | 6 |
| block | 2 | 1 | 0 | 0 | 4 | 25 | 2 | 1 | 0 | 7 | 2 | 104 |
