"""mkrefine.py KEY: concat labels/raw/parts/KEY_*.txt -> labels/raw/KEY.txt; gap_hit / missed_hit lines -> refine rows
(review/items/KEY__refine.json, src X, panels -0.3/-0.1/+0.1/+0.3 around the estimate)."""
import sys, json, glob
from pathlib import Path
B = Path('~/Desktop/RallyIQ/train_v11').expanduser(); k = sys.argv[1]
lines = []
for p in sorted(glob.glob(str(B / f'labels/raw/parts/{k}_*.txt'))):
    lines += [l.rstrip('\n') for l in open(p) if l.strip() and not l.startswith('#')]
def key(l):
    v = l.split('|'); return (int(v[0]), v[1].strip() == 'missed_hit')
lines.sort(key=key)
open(B / f'labels/raw/{k}.txt', 'w').write('# n | label | t | side | hitter | stroke | sure | reason\n' + '\n'.join(lines) + '\n')
D = json.load(open(B / f'review/items/{k}.json')); X = []
for l in lines:
    v = [x.strip() for x in l.split('|')]
    if v[1] in ('gap_hit', 'missed_hit'):
        X.append({'n': len(X), 'orig_n': int(v[0]), 'orig_t': float(v[2]), 't': float(v[2]), 'src': 'X',
                  'note': f"est {v[3]} {v[4]} {v[5]}", 'hitter': None, 'side': None, 'stroke': None})
json.dump({**D, 'items': X}, open(B / f'review/items/{k}__refine.json', 'w'), indent=0)
print(k, len(lines), 'lines', len(X), 'refine rows')
