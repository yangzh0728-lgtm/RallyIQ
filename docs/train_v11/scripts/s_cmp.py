"""s_cmp.py NEW_ROWS.pkl [--out RESID.json]: pre-registered S test. Residual (shipped recipe: rise/fall/oh/first, L2 0.003,
unsure rows weight 0.5). Arm A = fit on old rows (train_v9 n_rows.pkl, v9 labels) only; arm B = fit on old + new rows.
Both leave-one-video-out (a held-out video's rows never in the fit), scored on trusted rows of every video. Per-video bootstrap."""
import sys, os, json, pickle, math, random
import numpy as np
sys.path[:0] = [os.path.expanduser('~/Desktop/RallyIQ/train_v11/repo')]
from rallyiq.strokes import court as CM
cm = CM.get_model(); C = cm.classes; K = len(C); lam = 0.003; uw = 0.5; FEATS = ['rise', 'fall', 'oh', 'first']
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
        Gd = (Pq - Y) * ws[:, None]; gW = Gd.T @ Z + 2 * lam * W; gb = Gd.sum(0) + 2 * lam * b
        for j, (p, g) in enumerate(((W, gW), (b, gb))):
            m1[j] = 0.9 * m1[j] + 0.1 * g; m2[j] = 0.999 * m2[j] + 0.001 * g * g
            p -= 0.05 * (m1[j] / (1 - 0.9 ** it)) / (np.sqrt(m2[j] / (1 - 0.999 ** it)) + 1e-8)
    return W, b
old = pickle.load(open(os.path.expanduser('~/Desktop/RallyIQ/train_v9/ckpt/n_rows.pkl'), 'rb'))['rows']
new = pickle.load(open(sys.argv[1], 'rb'))['rows']
for r in old: r['_new'] = False
for r in new: r['_new'] = True
rows = [r for r in old + new if r['y'] in C]; isnew = np.asarray([r['_new'] for r in rows])
L, Z, y = prep(rows); w = np.asarray([1.0 if r['sure'] else uw for r in rows]); sure = np.asarray([bool(r['sure']) for r in rows])
keys = np.asarray([r['key'] for r in rows]); vids = sorted(set(keys))
pA = np.zeros(len(y), int); pB = np.zeros(len(y), int); base = L.argmax(1)
for v in vids:
    te = keys == v
    trA = ~te & ~isnew; W, b = fit(L[trA], Z[trA], y[trA], w[trA]); pA[te] = (L[te] + Z[te] @ W.T + b).argmax(1)
    trB = ~te; W, b = fit(L[trB], Z[trB], y[trB], w[trB]); pB[te] = (L[te] + Z[te] @ W.T + b).argmax(1)
def acc(p, m): return f"{np.mean(p[m] == y[m]):.3f} ({int(np.sum(p[m] == y[m]))}/{int(m.sum())})"
rng = random.Random(0)
def boot(m, pa, pb):
    vs = sorted(set(keys[m])); ds = []
    for _ in range(5000):
        ix = np.concatenate([np.where((keys == v) & m)[0] for v in (rng.choice(vs) for _ in vs)])
        ds.append(np.mean(pb[ix] == y[ix]) - np.mean(pa[ix] == y[ix]))
    ds.sort(); return f"[{ds[125]:+.3f}, {ds[4875]:+.3f}]"
res = {}
for nm, m in (('all trusted', sure), ('old trusted', sure & ~isnew), ('new trusted', sure & isnew)):
    line = f"{nm}: court_v2 {acc(base, m)} | A old-rows residual {acc(pA, m)} | B old+new {acc(pB, m)} | B-A CI {boot(m, pA, pB)}"
    print(line); res[nm] = line
print('rows', len(rows), 'trusted', int(sure.sum()), 'new trusted', int((sure & isnew).sum()), 'videos', len(vids))
if '--out' in sys.argv:
    out = sys.argv[sys.argv.index('--out') + 1]
    for arm, m in (('A', ~isnew), ('B', np.ones(len(y), bool))):
        W, b = fit(L[m], Z[m], y[m], w[m])
        json.dump({'kind': 'strokes_residual', 'classes': C, 'features': FEATS, 'W': W.round(5).tolist(), 'b': b.round(5).tolist(),
                   'meta': {'lam': lam, 'unsure_w': uw, 'rows': int(m.sum()), 'trusted': int((sure & m).sum()),
                            'source': "sam's own videos: train_v9 rows" + (' + train_v11 whole-video labels' if arm == 'B' else '')}},
                  open(out.replace('.json', f'_{arm}.json'), 'w'))
    print('wrote', out)
