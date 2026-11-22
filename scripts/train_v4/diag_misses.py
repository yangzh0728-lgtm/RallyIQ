"""Classify missed positives (FN) of a cached prediction set: serve (first labelled hit of a window or stroke *_serve),
untracked hitter (label hitter None on am), no candidate within 0.25 s (from the candidate cache), other."""
import sys, json, pickle, os
from pathlib import Path
from collections import Counter
B = Path('~/Desktop/RallyIQ/train_v4').expanduser(); sys.path[:0] = [str(B / 'repo'), str(B / 'repo/scripts')]
import amateur_hits as AH
name = sys.argv[1]; labels = Path(sys.argv[2] if len(sys.argv) > 2 else '~/Desktop/RallyIQ/train_v3/labels/amateur_v2.json').expanduser()
preds = json.load(open(B / f'ckpt/preds/{name}.json'))
truth = AH.load_truth(B / 'inputs', labels)
cands = {c.cid: c.t for c in pickle.load(open(os.path.expanduser('~/Desktop/RallyIQ/train_v3/ckpt/cands2/am_msc0.3.pkl'), 'rb'))}
cnt = Counter(); rows = []
for key, tr in truth.items():
    if key not in preds: continue
    pr = [p['t'] for p in preds[key] if any(a <= p['t'] <= b for a, b in tr['windows'])]
    pos = sorted([g for g in tr['labels'] if g['kind'] == 'pos'], key=lambda g: g['t'])
    used = set()
    for g in pos:
        m = [i for i, t in enumerate(pr) if i not in used and abs(t - g['t']) <= 0.25]
        if m:
            used.add(min(m, key=lambda i: abs(pr[i] - g['t']))); cnt['tp'] += 1; continue
        win = next((w for w in tr['windows'] if w[0] <= g['t'] <= w[1]), None)
        first = win is not None and not any(o['t'] < g['t'] and o['t'] >= win[0] for o in pos)
        serve = (g.get('stroke') or '').endswith('serve') or first
        untracked = tr['set'] == 'am7' and g.get('hitter') is None and g.get('status') in ('reviewed_sure', 'reviewed_unsure')
        ct = cands.get(key)
        nocand = ct is not None and (len(ct) == 0 or min(abs(x - g['t']) for x in ct) > 0.25)
        k = 'serve' if serve else ('untracked' if untracked else ('no_cand' if nocand else 'scored_low'))
        cnt[k] += 1; rows.append((key, round(g['t'], 2), k, g.get('stroke'), g.get('side')))
print(name, dict(cnt))
for r in rows: print('  ', *r)
# --- per-miss: nearest candidate and its LOVO probability (B5 models)
if len(sys.argv) > 3:
    from rallyiq.events.learned import get_scorer
    cl = {c.cid: c for c in pickle.load(open(os.path.expanduser('~/Desktop/RallyIQ/train_v3/ckpt/cands2/am_msc0.3.pkl'), 'rb'))}
    print('--- nearest candidate (dt, p) per miss')
    for key, t, k, st, side in rows:
        c = cl.get(key)
        if c is None or not len(c.t): continue
        sc = get_scorer(str(Path(sys.argv[3]).expanduser() / f'lovo_{key}.json'))
        import numpy as np
        d = np.abs(c.t - t); i = int(np.argmin(d))
        p = float(sc.predict([c.X[i]])[0])
        print('  ', key, t, k, st, side, f'dt={c.t[i] - t:+.2f} p={p:.2f}')
