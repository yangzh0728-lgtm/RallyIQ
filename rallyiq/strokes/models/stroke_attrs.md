# Shot attribute model (attrs_v1)

- trained on 1310 shots from 19 ShuttleSet matches
- held out: 191 shots from 6 other matches

| attribute | n (test) | accuracy | balanced accuracy | always-majority | recall per class |
|---|---|---|---|---|---|
| hand | 191 | 0.754 | 0.720 | 0.492 | around_head 0.54 (n 37), backhand 0.83 (n 60), forehand 0.79 (n 94) |
| contact | 191 | 0.838 | 0.821 | 0.602 | high 0.74 (n 76), low 0.90 (n 115) |
| direction | 153 | 0.935 | 0.928 | 0.614 | cross 0.90 (n 59), straight 0.96 (n 94) |

Pace reference speeds (m/s, 33rd / 67th percentile per family): {'block': [8.13, 9.72], 'clear': [8.01, 10.77], 'drive': [13.79, 16.49], 'drop': [9.88, 11.44], 'lift': [5.7, 7.4], 'net_kill': [14.56, 18.83], 'net_shot': [6.07, 7.65], 'push': [7.61, 10.03], 'serve': [8.64, 9.77], 'smash': [18.26, 22.6]}
