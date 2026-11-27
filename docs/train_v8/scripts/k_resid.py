"""Residual stroke stage: logits = log p(court x video) + W z + b, z = amateur shot features (rise, fall, overhead,
first + missing flags), W,b L2-regularised (classes unseen in the amateur rows stay at the base model).
Leave-one-video-out on amateur rows; also scores ShuttleSet test-clip rows with the all-amateur fit.
  court_resid.py LAM [unsure_w] [--out spec.json]"""
import sys, json, pickle, math
from pathlib import Path
from collections import Counter
import numpy as np
import os; B = Path('~/Desktop/RallyIQ/train_v8').expanduser(); sys.path[:0] = [str(B / 'repo')]
from rallyiq.strokes import court as CM
lam = float(sys.argv[1]); uw = float(sys.argv[2]) if len(sys.argv) > 2 and not sys.argv[2].startswith('--') else 0.5
out = sys.argv[sys.argv.index('--out') + 1] if '--out' in sys.argv else None
import os
FEATS = os.environ.get('FEATS', 'rise,fall,oh,first').split(',')
cm = CM.get_model(); C = cm.classes; K = len(C)
def prep(rows):
    X = np.asarray([r['X'] for r in rows], float); P = cm.proba(X); V = np.asarray([r['V'] for r in rows], float)
    have = ~np.isnan(V[:, 0]); G = P.copy()
    G[have] = np.sqrt(np.clip(P[have], 1e-6, None) * np.clip(V[have], 1e-6, None)); G /= G.sum(1, keepdims=True)
    fi = CM.FEATURES.index('first'); Z = []
    for i, r in enumerate(rows):
        z = []
        for f in FEATS:
            v = X[i, fi] if f == 'first' else r.get(f)
            miss = v is None or (isinstance(v, float) and math.isnan(v))
            z += [0.0 if miss else float(np.clip(v, -3, 3)), float(miss)]
        Z.append(z)
    y = np.asarray([C.index(r['y']) if r['y'] in C else -1 for r in rows])
    return np.log(np.clip(G, 1e-4, None)), np.asarray(Z), y
def fit(L, Z, y, w):
    W = np.zeros((K, Z.shape[1])); b = np.zeros(K); m1 = [np.zeros_like(W), np.zeros_like(b)]; m2 = [np.zeros_like(W), np.zeros_like(b)]
    Y = np.eye(K)[y]; ws = w / w.sum()
    for it in range(1, 3001):
        S = L + Z @ W.T + b; S -= S.max(1, keepdims=True); Pq = np.exp(S); Pq /= Pq.sum(1, keepdims=True)
        Gd = (Pq - Y) * ws[:, None]
        gW = Gd.T @ Z + 2 * lam * W; gb = Gd.sum(0) + 2 * lam * b
        for j, (p, g) in enumerate(((W, gW), (b, gb))):
            m1[j] = 0.9 * m1[j] + 0.1 * g; m2[j] = 0.999 * m2[j] + 0.001 * g * g
            p -= 0.05 * (m1[j] / (1 - 0.9 ** it)) / (np.sqrt(m2[j] / (1 - 0.999 ** it)) + 1e-8)
    return W, b
rows = pickle.load(open(B / os.environ.get('ROWS', 'ckpt/stroke_rows.pkl'), 'rb'))['rows']
L, Z, y = prep(rows); w = np.asarray([1.0 if r['sure'] else uw for r in rows]); sure = np.asarray([r['sure'] for r in rows])
pred = L.argmax(1).copy(); base = L.argmax(1)
for k in sorted({r['key'] for r in rows}):
    te = np.asarray([r['key'] == k for r in rows]); W, b = fit(L[~te], Z[~te], y[~te], w[~te])
    pred[te] = (L[te] + Z[te] @ W.T + b).argmax(1)
def acc(p, m, yy): return f"{np.mean(p[m] == yy[m]):.3f} ({int(np.sum(p[m] == yy[m]))}/{int(m.sum())})"
allm = np.ones(len(y), bool)
line = f"lam={lam} uw={uw}: amateur LOVO trusted {acc(base, sure, y)} -> {acc(pred, sure, y)}, all {acc(base, allm, y)} -> {acc(pred, allm, y)}"
import random
rng = random.Random(0); vids = sorted({r['key'] for r in rows}); kv = np.asarray([r['key'] for r in rows]); ds = []
for _ in range(3000):
    smp = [rng.choice(vids) for _ in vids]; ix = np.concatenate([np.where((kv == v) & sure)[0] for v in smp])
    ds.append(np.mean(pred[ix] == y[ix]) - np.mean(base[ix] == y[ix]))
ds.sort(); line += f" | trusted diff 95% CI over videos [{ds[75]:+.3f}, {ds[2925]:+.3f}]"
W, b = fit(L, Z, y, w)
ssp = B / os.environ.get('ROWS_SS', 'ckpt/stroke_rows_ss.pkl')
if ssp.exists():
    rs = pickle.load(open(ssp, 'rb'))['rows']; Ls, Zs, ys = prep(rs); m = ys >= 0
    line += f" | ShuttleSet test clips {acc(Ls.argmax(1), m, ys)} -> {acc((Ls + Zs @ W.T + b).argmax(1), m, ys)}"
print(line)
print('  amateur errors after', Counter((C[a], C[b_]) for a, b_, s in zip(y, pred, sure) if a != b_ and s).most_common(6))
with open(B / 'logs/court_resid.jsonl', 'a') as f: f.write(json.dumps({'lam': lam, 'uw': uw, 'line': line}) + '\n')
if out:
    Path(out).write_text(json.dumps({'kind': 'strokes_residual', 'classes': C, 'features': FEATS, 'W': W.round(5).tolist(), 'b': b.round(5).tolist(),
        'meta': {'lam': lam, 'unsure_w': uw, 'rows': len(rows), 'trusted': int(sure.sum()), 'source': "sam's own videos, train_v4 labels/amateur_v3.json"}}))
    print('wrote', out)
