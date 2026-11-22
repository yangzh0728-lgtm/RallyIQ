"""Fine-tune the court stroke model with amateur rows, leave-one-video-out.
  court_ft.py NAME [--aw 5] [--unsure 0.5] [--oh] [--no-video] [--out model.json]
Each amateur video is predicted by a model trained on ShuttleSet (same recipe as learn_court_strokes.py) + the
other amateur videos (weight aw per trusted row, aw*unsure per typed-unsure row). Prediction = court x video model
geometric mean (as in production), argmax. Baseline court_v2 is scored the same way."""
import sys, json, pickle, argparse, math, time
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
B = Path('~/Desktop/RallyIQ/train_v4').expanduser(); sys.path[:0] = [str(B / 'repo'), str(B / 'repo/scripts')]
import learn_court_strokes as LC
from rallyiq.strokes import court as CM
ap = argparse.ArgumentParser(); ap.add_argument('name'); ap.add_argument('--aw', type=float, default=5.0)
ap.add_argument('--unsure', type=float, default=0.5); ap.add_argument('--oh', action='store_true'); ap.add_argument('--rise', action='store_true'); ap.add_argument('--no-video', action='store_true')
ap.add_argument('--iters', type=int, default=150); ap.add_argument('--leaves', type=int, default=15); ap.add_argument('--out')
ap.add_argument('--folds', default='lovo')
a = ap.parse_args()
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
D = pickle.load(open(B / 'ckpt/stroke_rows.pkl', 'rb')); rows = D['rows']; C = D['classes']
cache = B / 'ckpt/ss_court_examples.pkl'
if cache.exists():
    Xs, ys = pickle.load(open(cache, 'rb'))
else:
    rallies = LC.load_rallies(B / 'inputs/shuttleset_set')
    parts = [LC.examples(rallies, np.random.default_rng(i), 0.4, 0.12, 0.0, d) for i, d in enumerate([0.0, 0.15, 0.3])]
    Xs = np.vstack([p[0] for p in parts]); ys = np.concatenate([p[1] for p in parts])
    pickle.dump((Xs, ys), open(cache, 'wb'))
def amX(rs):
    X = np.asarray([r['X'] for r in rs], float)
    if a.oh: X = np.hstack([X, np.asarray([[r['oh']] for r in rs], float)])
    if a.rise: X = np.hstack([X, np.asarray([[r['rise'], r['fall']] for r in rs], float)])
    return X
XsF = np.hstack([Xs, np.full((len(Xs), int(a.oh) + 2 * int(a.rise)), np.nan)])
def fit(extra):
    X = XsF; y = ys; w = np.ones(len(ys))
    if extra:
        X = np.vstack([X, amX(extra)]); y = np.concatenate([y, [r['y'] for r in extra]])
        w = np.concatenate([w, [a.aw * (1.0 if r['sure'] else a.unsure) for r in extra]])
    m = HGB(max_iter=a.iters, learning_rate=0.1, max_leaf_nodes=a.leaves, l2_regularization=1.0, random_state=0).fit(X, y, sample_weight=w)
    return m
def proba(m, rs):
    P = m.predict_proba(amX(rs)); out = np.zeros((len(rs), len(C)))
    for j, c in enumerate(m.classes_): out[:, C.index(c)] = P[:, j]
    return out
def combine(P, rs):
    if a.no_video: return P
    V = np.asarray([r['V'] for r in rs], float); have = ~np.isnan(V[:, 0])
    G = P.copy(); G[have] = np.sqrt(np.clip(P[have], 1e-6, None) * np.clip(V[have], 1e-6, None)); return G / G.sum(1, keepdims=True)
def score(pred, rs, tag):
    ok = [p == r['y'] for p, r in zip(pred, rs)]; tr = [o for o, r in zip(ok, rs) if r['sure']]
    conf = Counter((r['y'], p) for p, r in zip(pred, rs) if p != r['y'] and r['sure'])
    return {'tag': tag, 'all': round(float(np.mean(ok)), 3), 'n_all': len(ok), 'trusted': round(float(np.mean(tr)), 3), 'n_trusted': len(tr),
            'trusted_ok': int(sum(tr)), 'errors': conf.most_common(8)}
t0 = time.time()
keys = sorted({r['key'] for r in rows})
# baseline: bundled court_v2 (no amateur rows) - only when no extra feature
res = {}
if not a.oh and not a.rise:
    cm = CM.get_model(); Pb = cm.proba(np.asarray([r['X'] for r in rows], float))
    res['court_v2'] = score([C[i] for i in combine(Pb, rows).argmax(1)], rows, 'court_v2 bundled')
pred = [None] * len(rows)
for k in keys:
    tr = [r for r in rows if r['key'] != k]; te = [i for i, r in enumerate(rows) if r['key'] == k]
    m = fit(tr if a.aw > 0 else [])
    P = combine(proba(m, [rows[i] for i in te]), [rows[i] for i in te])
    for i, p in zip(te, P.argmax(1)): pred[i] = C[p]
    if a.aw == 0: break_after = True
res[a.name] = score(pred, rows, f'LOVO aw={a.aw} unsure={a.unsure} oh={a.oh} video={not a.no_video}')
per = defaultdict(lambda: [0, 0])
for p, r in zip(pred, rows):
    if r['sure']: per[r['key']][0] += p == r['y']; per[r['key']][1] += 1
res[a.name]['per_video_trusted'] = {k: v for k, v in per.items()}
print(json.dumps(res, indent=1)); print(f'{time.time() - t0:.0f}s')
with open(B / 'logs/court_ft.jsonl', 'a') as f: f.write(json.dumps(res) + '\n')
if a.out:
    from rallyiq.gbt import export_hgb
    m = fit(rows); spec = json.loads(CM.BUNDLED.read_text())
    spec.update(export_hgb(m)); spec['meta'] = {**spec['meta'], 'amateur_rows': len(rows), 'amateur_trusted': sum(r['sure'] for r in rows), 'aw': a.aw, 'unsure_w': a.unsure, 'oh': a.oh}
    spec['features'] = list(CM.FEATURES) + (['oh'] if a.oh else []) + (['rise', 'fall'] if a.rise else [])
    assert list(m.classes_) == C
    Path(a.out).write_text(json.dumps(spec, separators=(',', ':'))); print('wrote', a.out)
