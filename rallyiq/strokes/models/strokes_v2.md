# Stroke-type model (sklearn_hgb, features: v2, source: shuttleset)

rules_v1 on the same shots: accuracy 0.424

Shots: 915 from 18 groups, 5-fold grouped CV (out-of-fold)

- accuracy: 0.674
- macro F1: 0.560

| type | n | precision | recall | F1 |
| --- | --- | --- | --- | --- |
| short_serve | 41 | 0.72 | 0.93 | 0.81 |
| long_serve | 1 | 0.00 | 0.00 | 0.00 |
| clear | 71 | 0.90 | 0.79 | 0.84 |
| smash | 96 | 0.78 | 0.88 | 0.82 |
| drop | 51 | 0.54 | 0.55 | 0.54 |
| net_shot | 186 | 0.72 | 0.74 | 0.73 |
| cross_net | 53 | 0.62 | 0.49 | 0.55 |
| push | 80 | 0.36 | 0.26 | 0.30 |
| net_kill | 12 | 0.55 | 0.50 | 0.52 |
| drive | 29 | 0.24 | 0.21 | 0.22 |
| lift | 184 | 0.75 | 0.78 | 0.76 |
| block | 111 | 0.59 | 0.63 | 0.61 |

Confusion matrix (rows = truth, columns = predicted):

| | short_serve | long_serve | clear | smash | drop | net_shot | cross_net | push | net_kill | drive | lift | block |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| short_serve | 38 | 0 | 0 | 2 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| long_serve | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| clear | 0 | 0 | 56 | 6 | 5 | 2 | 0 | 2 | 0 | 0 | 0 | 0 |
| smash | 2 | 0 | 2 | 84 | 2 | 0 | 0 | 0 | 3 | 2 | 0 | 1 |
| drop | 1 | 0 | 4 | 8 | 28 | 4 | 1 | 0 | 0 | 0 | 0 | 5 |
| net_shot | 5 | 0 | 0 | 1 | 4 | 138 | 7 | 4 | 1 | 2 | 6 | 18 |
| cross_net | 3 | 0 | 0 | 2 | 3 | 12 | 26 | 1 | 0 | 0 | 1 | 5 |
| push | 1 | 0 | 0 | 1 | 0 | 5 | 0 | 21 | 0 | 7 | 42 | 3 |
| net_kill | 0 | 0 | 0 | 0 | 0 | 4 | 1 | 0 | 6 | 1 | 0 | 0 |
| drive | 0 | 0 | 0 | 4 | 3 | 1 | 0 | 2 | 0 | 6 | 0 | 13 |
| lift | 1 | 0 | 0 | 0 | 0 | 7 | 0 | 27 | 1 | 0 | 144 | 4 |
| block | 1 | 0 | 0 | 0 | 7 | 18 | 6 | 2 | 0 | 7 | 0 | 70 |
