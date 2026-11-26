"""Review sheets for hits + stroke types. Row per hit:
  [overview at contact: shuttle path from -0.3 s to the next hit (blue before, orange after, white dot per 0.2 s), hitter red]
  [3 hitter crops -0.20 / 0.00 / +0.15]  [overview at the next hit / +1.5 s]
render(items, out_prefix, rows_per_sheet=3, label_fn) ; item = {key, t, hitter, side, next_t, prev_t, title}"""
import cv2, json, gzip, os, sys, numpy as np
B = os.path.expanduser('~/Desktop/RallyIQ/train_v5'); I3 = os.path.expanduser('~/Desktop/RallyIQ/train_v3/inputs')
sys.path[:0] = [f'{B}/repo', f'{B}/repo/scripts']
import amateur_hits as AH
from pathlib import Path
CLIPS = json.load(open(f'{I3}/videos/index.json'))
RH = 330
def art(key):
    p = f'{B}/{os.environ.get("INP","inputs_B")}/new_runs/{key}'
    return p if os.path.exists(p) else str(AH.art_dir(Path(I3), key))
def rd(d, n):
    for p in (f'{d}/{n}.json', f'{d}/{n}.json.gz'):
        if os.path.exists(p): return json.load(gzip.open(p)) if p.endswith('.gz') else json.load(open(p))
class Video:
    def __init__(self, key):
        self.key = key; a = art(key)
        self.fps = rd(a, 'probe')['fps']
        self.pl = {f['frame']: f['players'] for f in rd(a, 'players')['frames']}
        self.sh = {p['frame']: p for p in rd(a, 'shuttle')['points']}
        src = next((s for s in (f'{B}/inputs_B/new_runs/{key}/source.mp4',) if os.path.exists(s)), None)
        self.cap = cv2.VideoCapture(src) if src else None; self.caps = {}; self.cache = {}
    def frame(self, t):
        f = round(t * self.fps)
        if f in self.cache: return self.cache[f]
        im = None
        if self.cap is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, f); ok, im = self.cap.read(); im = im if ok else None
        else:
            c = next((c for c in CLIPS.get(self.key, []) if c['t0'] <= t <= c['t1']), None)
            if c:
                cap = self.caps.get(c['file']) or cv2.VideoCapture(f"{I3}/videos/{c['file']}"); self.caps[c['file']] = cap
                cap.set(cv2.CAP_PROP_POS_FRAMES, round((t - c['t0']) * cap.get(cv2.CAP_PROP_FPS))); ok, im = cap.read(); im = im if ok else None
        if len(self.cache) > 300: self.cache.clear()
        self.cache[f] = im
        return im
    def players(self, f):
        for d in (0, -1, 1, -2, 2, -3, 3):
            if f + d in self.pl: return self.pl[f + d]
        return []
def crop(im, box, ar, h):
    x0, y0, x1, y1 = box; cx, cy = (x0 + x1) / 2, (y0 + y1) / 2; bh = max(y1 - y0, (x1 - x0) / ar); bw = bh * ar
    x0, x1, y0, y1 = int(cx - bw / 2), int(cx + bw / 2), int(cy - bh / 2), int(cy + bh / 2)
    P = 3000; pad = cv2.copyMakeBorder(im, P, P, P, P, cv2.BORDER_CONSTANT)
    return cv2.resize(pad[y0 + P:y1 + P, x0 + P:x1 + P], (int(h * ar), h))
def boxes(v, im, t, hitter, side):
    f = round(t * v.fps); im = im.copy()
    for p in v.players(f):
        b = list(map(int, p['bbox'])); me = p['player_id'] == hitter or (hitter is None and p.get('side') == side)
        col = (0, 0, 255) if me else (255, 200, 0)
        cv2.rectangle(im, b[:2], b[2:], col, 3 if me else 2); cv2.putText(im, f"{p['player_id']}{(p.get('side') or '')[:1]}", (b[0], b[1] - 6), 0, 1.0, col, 2)
    return im
def flight(v, t0, t1, tc):
    """Track points of the shuttle hit at tc only: the run of points at the hit, carried over gaps while the next run starts
    where the last one was heading. A run that starts elsewhere (another shuttle) is cut off."""
    f0, f1, fc = round(t0 * v.fps), round(t1 * v.fps), round(tc * v.fps)
    pts = [(f, v.sh[f]['x'], v.sh[f]['y']) for f in range(f0, f1 + 1) if v.sh.get(f) and v.sh[f].get('x') is not None]
    runs = []
    for p in pts:
        if runs:
            q = runs[-1][-1]; df = p[0] - q[0]
            if df <= max(2, 0.08 * v.fps) and np.hypot(p[1] - q[1], p[2] - q[2]) <= 0.15 * 2203 * df * 30 / v.fps: runs[-1].append(p); continue
        runs.append([p])
    if not runs: return []
    i0 = min(range(len(runs)), key=lambda i: 0 if runs[i][0][0] <= fc <= runs[i][-1][0] else min(abs(runs[i][0][0] - fc), abs(runs[i][-1][0] - fc)))
    keep = [runs[i0]]
    for step in (1, -1):
        cur = runs[i0]; i = i0 + step
        while 0 <= i < len(runs):
            r = runs[i]; a, b = (cur[-1], r[0]) if step > 0 else (r[-1], cur[0]); df = max(b[0] - a[0], 1)
            e = (cur[-3:] if step > 0 else cur[:3]); ve = np.array(e[-1][1:]) - np.array(e[0][1:]); ve = ve / max(e[-1][0] - e[0][0], 1)
            pred = np.array(cur[-1][1:]) + ve * df if step > 0 else np.array(cur[0][1:]) - ve * df
            tgt = np.array(b[1:]) if step > 0 else np.array(a[1:])
            if np.linalg.norm(pred - tgt) <= 90 + 0.7 * np.linalg.norm(ve) * df and df < 0.8 * v.fps: keep.append(r); cur = r
            i += step
    return sorted(p for r in keep for p in r)
def path(v, im, t0, t1, tc):
    for f, x, y in flight(v, t0, t1, tc):
        dt = f / v.fps - tc; col = (255, 120, 0) if dt < 0 else (0, 110, 255)
        cv2.circle(im, (int(x), int(y)), 5, col, -1)
        if abs(dt * 5 - round(dt * 5)) < 2.5 / v.fps and dt > 0: cv2.circle(im, (int(x), int(y)), 9, (255, 255, 255), 2)
    p = v.sh.get(round(tc * v.fps))
    if p and p.get('x') is not None: cv2.circle(im, (int(p['x']), int(p['y'])), 18, (0, 255, 0), 3)
    return im
def ov(v, im, label, w):
    h = int(w * 9 / 16); c = cv2.resize(im, (w, h)); cv2.putText(c, label, (6, h - 10), 0, 0.8, (0, 255, 255), 2); return c
def row(v, it):
    t = it['t']; nt = it.get('next_t') or t + 1.5; nt = min(nt, t + 2.5)
    base = v.frame(t); base = np.zeros((1080, 1920, 3), np.uint8) if base is None else base
    o1 = path(v, boxes(v, base, t, it.get('hitter'), it.get('side')), t - 0.3, nt, t)
    W1 = int(RH * 16 / 9)
    nx = v.frame(nt); nx = np.zeros_like(base) if nx is None else path(v, boxes(v, nx, nt, None, None), t, nt + 0.3, nt)
    pls = v.players(round(t * v.fps)); me = [p for p in pls if p['player_id'] == it.get('hitter')] or [p for p in pls if it.get('hitter') is None and p.get('side') == it.get('side')]
    sp = v.sh.get(round(t * v.fps))
    hb = me[0]['bbox'] if me else ([sp['x'] - 80, sp['y'] - 80, sp['x'] + 80, sp['y'] + 160] if sp and sp.get('x') is not None else [760, 340, 1160, 740])
    bw, bh = hb[2] - hb[0], hb[3] - hb[1]; hb = [hb[0] - bw * 0.8, hb[1] - bh * 0.5, hb[2] + bw * 0.8, hb[3] + bh * 0.1]
    parts = [ov(v, o1, 'AT HIT  path ->next', W1)]
    for dt in (-0.2, 0.0, 0.15):
        im = v.frame(t + dt); im = np.zeros_like(base) if im is None else im.copy()
        q = v.sh.get(round((t + dt) * v.fps))
        if q and q.get('x') is not None: cv2.circle(im, (int(q['x']), int(q['y'])), 12, (0, 255, 0), 2)
        c = crop(im, hb, 0.7, RH); cv2.putText(c, f'{dt:+.2f}', (4, RH - 8), 0, 0.8, (255, 255, 255), 2); parts.append(c)
    parts.append(ov(v, nx, f"NEXT {nt - t:+.2f}s", W1))
    r = cv2.hconcat([cv2.resize(p, (p.shape[1], RH)) for p in parts])
    hdr = np.zeros((36, r.shape[1], 3), np.uint8); cv2.putText(hdr, it['title'], (6, 26), 0, 0.8, (0, 255, 255), 2)
    return cv2.vconcat([hdr, r])
def render(items, out_dir, prefix, per=3):
    os.makedirs(out_dir, exist_ok=True); vids = {}; rows = []; files = []
    for it in items:
        v = vids.get(it['key']) or Video(it['key']); vids[it['key']] = v
        rows.append(row(v, it))
    for k in range(0, len(rows), per):
        fn = f'{out_dir}/{prefix}_{k // per:02d}.jpg'; cv2.imwrite(fn, cv2.vconcat(rows[k:k + per]), [cv2.IMWRITE_JPEG_QUALITY, 78]); files.append(fn)
    return files
