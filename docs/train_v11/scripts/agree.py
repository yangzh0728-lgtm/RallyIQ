"""agree.py: pass-1 (labels/raw) vs pass-2 (labels/pass2) agreement on the 10% sample."""
import glob, json
from pathlib import Path
from collections import Counter
def parse(p):
    d = {}
    for l in open(p):
        v = [x.strip() for x in l.split('|')]
        if len(v) < 8 or not v[0].isdigit() or v[1] == 'missed_hit': continue
        d[int(v[0])] = v
    return d
def cls(v):
    lab = v[1]
    if lab.startswith('dup'): return 'hit'   # duplicate of a hit on another row = the contact exists
    if lab in ('hit', 'gap_hit'): return 'unsure' if 'unsure:hit' in v[6] else 'hit'
    if lab in ('not_a_hit', 'gap_none'): return 'unsure' if 'unsure:hit' in v[6] else 'no'
    return 'unsure'
T = Counter(); dis = []
for p in sorted(glob.glob('labels/pass2/*.txt')):
    k = Path(p).stem; a = parse(f'labels/raw/{k}.txt'); b = parse(p)
    for n, vb in b.items():
        va = a[n]; ca, cb = cls(va), cls(vb); T['rows'] += 1
        if 'unsure' in (ca, cb): T['either_unsure'] += 1; continue
        T['decided'] += 1; T['hit_agree'] += ca == cb
        if ca != cb: dis.append((k, n, va[1], va[2], vb[1], vb[2], va[7][:60], vb[7][:60]))
        if ca == cb == 'hit':
            T['both_hit'] += 1; T['t_close'] += abs(float(va[2]) - float(vb[2])) <= .15
            T['side_agree'] += va[3] == vb[3]
            if va[4] not in ('none', '-') and vb[4] not in ('none', '-'): T['hitter_n'] += 1; T['hitter_agree'] += va[4] == vb[4]
            if va[5] != 'unknown' and vb[5] != 'unknown':
                T['stroke_n'] += 1; T['stroke_agree'] += va[5] == vb[5]
                if 'unsure:stroke' not in va[6] and 'unsure:stroke' not in vb[6]: T['stroke_sure_n'] += 1; T['stroke_sure_agree'] += va[5] == vb[5]
print(dict(T))
for d in dis: print(d)
