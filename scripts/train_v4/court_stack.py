"""Second stage on top of court_v2 x video model: multinomial logistic regression on [log p (12 classes), rise, fall,
overhead, first-in-rally] trained on amateur rows only, leave-one-video-out.  court_stack.py [C] [unsure_w] [feats]"""
import sys, json, pickle, math
from pathlib import Path
from collections import Counter
import numpy as np
B = Path('~/Desktop/RallyIQ/train_v4').expanduser(); sys.path[:0] = [str(B / 'repo')]
from rallyiq.strokes import court as CM
from sklearn.linear_model import LogisticRegression
Creg = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
uw = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
feats = sys.argv[3].split(',') if len(sys.argv) > 3 else ['rise', 'fall', 'oh', 'first']
D = pickle.load(open(B / 'ckpt/stroke_rows.pkl', 'rb')); rows = D['rows']; C = D['classes']
cm = CM.get_model(); X = np.asarray([r['X'] for r in rows], float)
P = cm.proba(X); V = np.asarray([r['V'] for r in rows], float); have = ~np.isnan(V[:, 0])
G = P.copy(); G[have] = np.sqrt(np.clip(P[have], 1e-6, None) * np.clip(V[have], 1e-6, None)); G /= G.sum(1, keepdims=True)
fi = CM.FEATURES.index('first')
def z(i):
    r = rows[i]; extra = []
    for f in feats:
        v = X[i, fi] if f == 'first' else r[f]
        extra += [0.0 if (v is None or math.isnan(v)) else float(np.clip(v, -3, 3)), float(v is None or math.isnan(v))]
    return np.concatenate([np.log(np.clip(G[i], 1e-4, None)), extra])
Z = np.asarray([z(i) for i in range(len(rows))]); y = np.asarray([r['y'] for r in rows]); w = np.asarray([1.0 if r['sure'] else uw for r in rows])
keys = sorted({r['key'] for r in rows}); pred = np.asarray([C[k] for k in G.argmax(1)], dtype=object); base = pred.copy()
for k in keys:
    tr = np.asarray([r['key'] != k for r in rows]); te = ~tr
    m = LogisticRegression(C=Creg, max_iter=2000).fit(Z[tr], y[tr], sample_weight=w[tr])
    pred[te] = m.predict(Z[te])
sure = np.asarray([r['sure'] for r in rows])
def acc(p, m): return f"{np.mean(p[m] == y[m]):.3f} ({int(np.sum(p[m] == y[m]))}/{int(m.sum())})"
print(f"C={Creg} uw={uw} feats={feats}: base trusted {acc(base, sure)} all {acc(base, np.ones(len(y), bool))} | stacked trusted {acc(pred, sure)} all {acc(pred, np.ones(len(y), bool))}")
print('  errors', Counter((a, b) for a, b, s in zip(y, pred, sure) if a != b and s).most_common(6))
