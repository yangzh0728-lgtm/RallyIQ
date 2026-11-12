# Shot attribute model (attrs_v2)

- trained on 2515 shots from 34 ShuttleSet matches
- held out: 191 shots from 6 other matches

| attribute | n (test) | accuracy | balanced accuracy | always-majority | recall per class |
|---|---|---|---|---|---|
| hand | 191 | 0.775 | 0.753 | 0.492 | around_head 0.62 (n 37), backhand 0.85 (n 60), forehand 0.79 (n 94) |
| contact | 191 | 0.827 | 0.816 | 0.602 | high 0.76 (n 76), low 0.87 (n 115) |
| direction | 153 | 0.935 | 0.928 | 0.614 | cross 0.90 (n 59), straight 0.96 (n 94) |

Pace reference speeds (m/s, 33rd / 67th percentile per family): {'block': [7.84, 9.6], 'clear': [8.26, 10.06], 'drive': [12.48, 16.5], 'drop': [9.88, 11.43], 'lift': [5.73, 7.44], 'net_kill': [14.27, 18.83], 'net_shot': [5.93, 7.6], 'push': [7.55, 9.76], 'serve': [8.63, 9.77], 'smash': [17.76, 22.13]}
