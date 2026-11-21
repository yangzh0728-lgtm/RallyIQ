"""Paired bootstrap over videos of two amateur_hits.py reports (--json outputs): F1 difference B - A with a 95% interval.

    python scripts/boot_compare.py A.json B.json [--n 5000]
"""
import argparse
import json
import random

QA6 = ("own_20260529_0006", "own_20260529_0003", "own_20260529_0004", "own_20260529_0008", "own_20260906_0615", "own_20260906_0620")


def f1(rows):
    tp = sum(r["tp"] for r in rows); fp = sum(r["fp"] for r in rows); pos = sum(r["pos"] for r in rows)
    return 2 * tp / max(2 * tp + fp + (pos - tp), 1)


ap = argparse.ArgumentParser()
ap.add_argument("a"); ap.add_argument("b"); ap.add_argument("--n", type=int, default=5000)
x = ap.parse_args()
A = json.load(open(x.a))["per_video"]; B = json.load(open(x.b))["per_video"]
rng = random.Random(0)
for name, vids in (("qa6", [v for v in A if v in QA6]), ("am7", [v for v in A if v not in QA6]), ("all", list(A))):
    d0 = f1([B[v] for v in vids]) - f1([A[v] for v in vids])
    ds = []
    for _ in range(x.n):
        s = [rng.choice(vids) for _ in vids]
        ds.append(f1([B[v] for v in s]) - f1([A[v] for v in s]))
    ds.sort()
    lo, hi = ds[int(0.025 * x.n)], ds[int(0.975 * x.n)]
    print(f"{name}: F1 {f1([A[v] for v in vids]):.3f} -> {f1([B[v] for v in vids]):.3f}  diff {d0:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  P(diff>0) {sum(d > 0 for d in ds) / x.n:.2f}")
