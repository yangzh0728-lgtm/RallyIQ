"""Paired bootstrap over videos on cached predictions: boot.py LABELS A B  (F1 B - A, 95% CI) for qa6 / other / all / r4."""
import sys, json, random
from pathlib import Path
B = Path('~/Desktop/RallyIQ/train_v4').expanduser(); sys.path[:0] = [str(B / 'repo'), str(B / 'repo/scripts')]
import amateur_hits as AH
R4 = {'own_20260906_0617', 'own_20260906_0618', 'own_20260906_0619', 'own_20260906_0621'}
clips = AH.load_clips(B / 'inputs', Path(sys.argv[1]).expanduser())
pa, pb = (json.load(open(B / f'ckpt/preds/{n}.json')) for n in sys.argv[2:4])
A = {c.cid: AH.score_clip(c, pa[c.cid]) for c in clips}; Bv = {c.cid: AH.score_clip(c, pb[c.cid]) for c in clips}
def f1(rows):
    tp = sum(r['tp'] for r in rows); fp = sum(r['fp'] for r in rows); pos = sum(r['pos'] for r in rows)
    return 2 * tp / max(2 * tp + fp + (pos - tp), 1)
rng = random.Random(0); N = 5000
sets = {'qa6': [c.cid for c in clips if c.truth['set'] == 'qa6'], 'am (other)': [c.cid for c in clips if c.truth['set'] != 'qa6'],
        'r4 (new videos)': [c.cid for c in clips if c.cid in R4], 'all': [c.cid for c in clips]}
for name, vids in sets.items():
    if not vids: continue
    d0 = f1([Bv[v] for v in vids]) - f1([A[v] for v in vids]); ds = []
    for _ in range(N):
        s = [rng.choice(vids) for _ in vids]; ds.append(f1([Bv[v] for v in s]) - f1([A[v] for v in s]))
    ds.sort()
    print(f"{sys.argv[2]} -> {sys.argv[3]} {name:16s} ({len(vids)} videos): {f1([A[v] for v in vids]):.3f} -> {f1([Bv[v] for v in vids]):.3f}  "
          f"diff {d0:+.3f}  95% CI [{ds[int(.025 * N)]:+.3f}, {ds[int(.975 * N)]:+.3f}]  P(>0) {sum(d > 0 for d in ds) / N:.2f}")
