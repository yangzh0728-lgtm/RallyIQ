"""items.py KEY ART_DIR: review rows for a whole video (window = full length).
Sources (merged within 0.2 s): P = production pipeline hits (own settings), L = scorer at low threshold (0.1, short rallies
kept), R = rules candidates (scorer off, no swing veto), G = gap probe (> 1.6 s without any row inside a rally of the L run,
or > 1.0 s between a rally start and its first row). Stroke guess from the production strokes stage when present.
-> review/items/<KEY>.json"""
import sys, os, json, gzip
from pathlib import Path
from dataclasses import replace
R = Path('~/Desktop/RallyIQ/train_v11/repo').expanduser(); sys.path[:0] = [str(R), str(R / 'scripts')]
import amateur_hits as AH
from rallyiq.events.config import EventsConfig
B = Path('~/Desktop/RallyIQ/train_v11').expanduser()
k, art = sys.argv[1], Path(sys.argv[2])
dur = json.loads(AH.read_json(art, 'probe'))['duration_ms'] / 1000
c = AH.AmClip(k, art, {'set': 'v11', 'windows': [(0, dur)], 'labels': []})
base = {**AH.AMATEUR_BASE}
P = c.predict_full(replace(EventsConfig(), **base))
Lcfg = replace(EventsConfig(), **{**base, 'min_candidate_conf': 0.1, 'keep_threshold': 0.1, 'min_no_hit_rally_ms': 0.0})
L = c.predict_full(Lcfg)
Rr = c.predict_full(replace(EventsConfig(), **{**base, 'hit_scorer': 'off', 'swing_veto': 0.0, 'min_no_hit_rally_ms': 0.0}))
rallies = [(r.start_ms / 1000, r.end_ms / 1000) for r in c.run(Lcfg)]
# production stroke guesses by frame
st = {}
sj = AH.read_json(art, 'strokes')
if sj:
    for s in json.loads(sj).get('shots', []):
        st[s['frame']] = (s.get('stroke') or {}).get('shot_type')
fps = c.fps
items = []
for src, lst in (('P', P), ('L', L), ('R', Rr)):
    for p in lst:
        m = next((it for it in items if abs(it['t'] - p['t']) <= .2), None)
        if m:
            if src not in m['src']: m['src'] += src
            continue
        f = round(p['t'] * fps)
        g = min(st, key=lambda q: abs(q - f)) if st else None
        items.append({'t': round(p['t'], 3), 'src': src, 'hitter': p['hitter'], 'side': p['side'], 'conf': p.get('conf'),
                      'stroke': st[g] if (g is not None and abs(g - f) <= 3 and src == 'P') else None})
items.sort(key=lambda x: x['t'])
gaps = []
for a, b in rallies:
    ts = [it['t'] for it in items if a - .2 <= it['t'] <= b + .2]
    if not ts:
        if b - a > 1.0: gaps.append((a, b))
        continue
    if ts[0] - a > 1.0: gaps.append((a, ts[0]))
    for x, y in zip(ts, ts[1:]):
        if y - x > 1.6: gaps.append((x, y))
for a, b in gaps:
    items.append({'t': round((a + b) / 2, 3), 'src': 'G', 'gap': [round(a, 2), round(b, 2)], 'hitter': None, 'side': None, 'stroke': None})
items.sort(key=lambda x: x['t'])
for i, it in enumerate(items): it['n'] = i
(B / 'review/items').mkdir(parents=True, exist_ok=True)
json.dump({'key': k, 'art': str(art), 'fps': fps, 'duration': dur, 'rallies': rallies, 'items': items},
          open(B / f'review/items/{k}.json', 'w'), indent=0)
from collections import Counter
print(k, 'items', len(items), Counter(it['src'] for it in items), 'P', len(P), 'gaps', len(gaps), 'rallies', len(rallies))
