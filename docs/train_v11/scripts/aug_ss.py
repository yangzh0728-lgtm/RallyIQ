"""aug_ss.py SRC_SS_DIR OUT_DIR [copies=2]: C (PLAN.md addendum). Synthetic copies of ShuttleSet train clips: exact camera
rotation homography on every image-space quantity + own-footage-like shuttle detection loss. Writes OUT_DIR/index.json
(original clips + copies, artifacts as absolute paths), OUT_DIR/labels/*.json and OUT_DIR/data/<cid>_aN/artifacts."""
import sys, json, math, shutil, random
from pathlib import Path
import numpy as np
src, out = Path(sys.argv[1]).expanduser(), Path(sys.argv[2]).expanduser(); NC = int(sys.argv[3]) if len(sys.argv) > 3 else 2
(out / 'labels').mkdir(parents=True, exist_ok=True); (out / 'data').mkdir(exist_ok=True)


def rot(yaw, pitch, roll):
    y, p, r = map(math.radians, (yaw, pitch, roll))
    Ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    Rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    Rz = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])
    return Rz @ Rx @ Ry


def img_h(w, h, rng):
    f = 0.9 * w; K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
    R = rot(rng.uniform(-15, 15), rng.uniform(-10, 10), rng.uniform(-3, 3)); M = K @ R @ np.linalg.inv(K)
    return M / M[2, 2]


def tp(M, x, y):
    q = M @ np.array([x, y, 1.0]); return float(q[0] / q[2]), float(q[1] / q[2])


def transform(art, dst, M, w, h, rng):
    dst.mkdir(parents=True, exist_ok=True)
    for n in ('probe.json', 'events.json', 'strokes.json'):
        if (art / n).exists(): shutil.copy(art / n, dst / n)
    Mi = np.linalg.inv(M)
    cal = json.load(open(art / 'calibration.json'))
    for s in cal.get('segments', []):
        if s.get('homography'): Hn = np.array(s['homography']) @ Mi; s['homography'] = (Hn / Hn[2, 2]).tolist()
        if s.get('image_corners'): s['image_corners'] = [list(tp(M, *p)) for p in s['image_corners']]
        s['other_courts'] = [[list(tp(M, *p)) for p in c] for c in s.get('other_courts', [])]
        for kp in s.get('keypoints', []):
            if kp.get('image'): kp['image'] = list(tp(M, *kp['image']))
    json.dump(cal, open(dst / 'calibration.json', 'w'))
    pl = json.load(open(art / 'players.json'))
    for fr in pl['frames']:
        keep = []
        for p in fr['players']:
            b = p['bbox']; cs = [tp(M, x, y) for x, y in ((b[0], b[1]), (b[2], b[1]), (b[0], b[3]), (b[2], b[3]))]
            p['bbox'] = [min(c[0] for c in cs), min(c[1] for c in cs), max(c[0] for c in cs), max(c[1] for c in cs)]
            if p.get('keypoints'): p['keypoints'] = [[*tp(M, k[0], k[1]), *k[2:]] for k in p['keypoints']]
            if p.get('foot_px'): p['foot_px'] = list(tp(M, *p['foot_px']))
            fx, fy = p['foot_px'] if p.get('foot_px') else ((p['bbox'][0] + p['bbox'][2]) / 2, p['bbox'][3])
            if -0.05 * w <= fx <= 1.05 * w and fy <= 1.05 * h: keep.append(p)
        fr['players'] = keep
    json.dump(pl, open(dst / 'players.json', 'w'))
    sh = json.load(open(art / 'shuttle.json')); pts = sh['points']; frames = sorted({p['frame'] for p in pts})
    drop = set(); f0 = frames[0] if frames else 0; f1 = frames[-1] if frames else -1; f = f0
    while f <= f1:
        if rng.random() < 0.03:
            L = rng.randint(2, 10); drop.update(range(f, f + L)); f += L
        else: f += 1
    sig = 2.0 * w / 1920; outp = []
    for p in pts:
        if p['frame'] in drop or rng.random() < 0.10: continue
        x, y = tp(M, p['x'], p['y'])
        if not (0 <= x < w and 0 <= y < h): continue
        p['x'] = round(x + rng.gauss(0, sig), 2); p['y'] = round(y + rng.gauss(0, sig), 2); outp.append(p)
    sh['points'] = outp; json.dump(sh, open(dst / 'shuttle.json', 'w'))
    return len(pts), len(outp)


idx = json.load(open(src / 'index.json')); new = {}
rng = random.Random(11)
for cid, v in idx.items():
    new[cid] = v; shutil.copy(src / 'labels' / f'{cid}.json', out / 'labels' / f'{cid}.json') if (src / 'labels' / f'{cid}.json').exists() else None
    if v.get('state') != 'succeeded': continue
    art = Path(v['artifacts']); pr = json.load(open(art / 'probe.json')); w, h = pr['width'], pr['height']
    for a in range(NC):
        M = img_h(w, h, rng); c2 = f'{cid}_a{a}'; dst = out / 'data' / c2 / 'artifacts'
        n0, n1 = transform(art, dst, M, w, h, rng)
        new[c2] = {**v, 'artifacts': str(dst), 'aug_of': cid, 'aug_H': M.round(6).tolist()}
        shutil.copy(src / 'labels' / f'{cid}.json', out / 'labels' / f'{c2}.json')
        print(c2, 'shuttle points', n0, '->', n1, flush=True)
json.dump(new, open(out / 'index.json', 'w'), indent=0)
print('clips', len(new))
