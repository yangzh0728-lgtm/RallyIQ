"""Round 4b: more rallies (3-16 v3b hits) in 0617 and 0619 (singles play, clear camera), outside the round-4 windows."""
import sys, json, random
from dataclasses import replace
from pathlib import Path
B = Path('~/Desktop/RallyIQ/train_v4').expanduser()
sys.path[:0] = [str(B / 'repo'), str(B / 'repo/scripts')]
import amateur_hits as AH
from rallyiq.events.config import EventsConfig
v2p = str(B / 'repo/rallyiq/events/models/hit_scorer_v2.json')
old = json.load(open(B / 'labels/r4/candidates.json'))['windows']
rng = random.Random(42)
out = {'windows': {}, 'cands': {}}
for key, n_r in (('own_20260906_0618', 5), ('own_20260906_0621', 4)):
    c = AH.AmClip(key, B / 'inputs/new_runs' / key, {'set': 'am7', 'windows': [], 'labels': []})
    cfg3 = replace(EventsConfig(), **AH.AMATEUR_BASE); cfg2 = replace(EventsConfig(), **{**AH.AMATEUR_BASE, 'hit_scorer': v2p})
    rs = c.run(cfg3)
    ok = [r for r in rs if 3 <= len(r.hits) <= 16 and not any(a - 2 <= r.end_ms / 1000 and r.start_ms / 1000 <= b + 2 for a, b in old[key])]
    pick = []
    for r in rng.sample(ok, len(ok)):
        if len(pick) < n_r and not any(r.start_ms <= q.end_ms + 2000 and q.start_ms <= r.end_ms + 2000 for q in pick):
            pick.append(r)
    pick.sort(key=lambda r: r.start_ms)
    wins = [(round(r.start_ms / 1000 - 0.5, 2), round(r.end_ms / 1000 + 0.5, 2)) for r in pick]
    p3, p2 = c.predict_full(cfg3), c.predict_full(cfg2)
    rows = []
    for a, b in wins:
        h3 = [dict(p, src='v3b') for p in p3 if a <= p['t'] <= b]
        for h in [dict(p, src='v2') for p in p2 if a <= p['t'] <= b]:
            m = [x for x in h3 if abs(x['t'] - h['t']) <= 0.15]
            if m: m[0]['src'] = 'both'
            else: h3.append(h)
        rows += sorted(h3, key=lambda x: x['t'])
    out['windows'][key] = wins; out['cands'][key] = rows
    print(key, len(ok), 'eligible; windows', wins, '->', len(rows), flush=True)
(B / 'labels/r4c').mkdir(exist_ok=True)
json.dump(out, open(B / 'labels/r4c/candidates.json', 'w'), indent=1)
