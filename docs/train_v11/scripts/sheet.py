"""sheet.py KEY [N0 N1] : contact sheets for review/items/<KEY>.json -> review/sheets/<KEY>/<KEY>_NNN.jpg (6 rows each).
Row = 4 panels. Candidate rows: -0.2 / 0 / +0.2 / +0.4 s around the row time. Gap rows (src G): 4 times evenly inside the gap.
Panels: player boxes (green = near, orange = far) with the pipeline id, shuttle points of the last 0.25 s (yellow),
shuttle at the panel time (cyan ring). Crop = players + shuttle around the row, 16:9."""
import sys, os, json, gzip, cv2, numpy as np
from pathlib import Path
B = Path('~/Desktop/RallyIQ/train_v11').expanduser()
k = sys.argv[1]; SUF = os.environ.get('SUFFIX', ''); D = json.load(open(B / f'review/items/{k}{SUF}.json')); art = Path(D['art']); fps = D['fps']
def rd(n):
    for p in (art / f'{n}.json', art / f'{n}.json.gz'):
        if p.exists(): return json.load(gzip.open(p, 'rt') if p.suffix == '.gz' else open(p))
S = {p['frame']: p for p in rd('shuttle')['points']}
PL = {f['frame']: f['players'] for f in rd('players')['frames']}
PLk = sorted(PL)
import bisect
def players_near(f):
    i = bisect.bisect_left(PLk, f)
    best = min((PLk[j] for j in (i - 1, i) if 0 <= j < len(PLk)), key=lambda g: abs(g - f), default=None)
    return PL[best] if best is not None and abs(best - f) <= 4 else []
cap = cv2.VideoCapture(str(B / f'src/{k}.mp4')); W, Hh = int(cap.get(3)), int(cap.get(4))
def grab(f):
    cap.set(1, max(f, 0)); ok, im = cap.read(); return im if ok else np.zeros((Hh, W, 3), np.uint8)
def cbox(f0, f1):
    xs, ys = [], []
    for f in range(f0, f1 + 1, 2):
        for p in players_near(f):
            b = p['bbox']; xs += [b[0], b[2]]; ys += [b[1], b[3]]
        if f in S: xs.append(S[f]['x']); ys.append(S[f]['y'])
    if not xs: return (0, 0, W, Hh)
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    w = min(W, max(560, (max(xs) - min(xs)) * 1.2, (max(ys) - min(ys)) * 1.2 * 16 / 9)); h = w * 9 / 16
    x0 = int(min(max(cx - w / 2, 0), W - w)); y0 = int(min(max(cy - h / 2, 0), Hh - h))
    return (x0, y0, int(x0 + w), int(y0 + h))
def panel(f, box, title, im):
    for p in players_near(f):
        b = list(map(int, p['bbox'])); col = (0, 200, 0) if p.get('side') == 'near' else (0, 140, 255)
        cv2.rectangle(im, b[:2], b[2:], col, 2); cv2.putText(im, p['player_id'], (b[0], b[1] - 6), 0, 1.0, col, 2)
    for g in range(f - round(.25 * fps), f):
        if g in S: cv2.circle(im, (int(S[g]['x']), int(S[g]['y'])), 4, (0, 255, 255), -1)
    if f in S: cv2.circle(im, (int(S[f]['x']), int(S[f]['y'])), 14, (255, 255, 0), 2)
    x0, y0, x1, y1 = box; im = cv2.resize(im[y0:y1, x0:x1], (480, 270))
    cv2.rectangle(im, (0, 0), (479, 269), (40, 40, 40), 1)
    cv2.putText(im, title, (5, 22), 0, .6, (0, 0, 0), 4); cv2.putText(im, title, (5, 22), 0, .6, (255, 255, 255), 1)
    return im
items = D['items']
n0 = int(sys.argv[2]) if len(sys.argv) > 2 else 0; n1 = int(sys.argv[3]) if len(sys.argv) > 3 else len(items)
od = B / f'review/sheets/{k}{SUF}'; od.mkdir(parents=True, exist_ok=True)
plan = []  # (row, title cells, box)
for it in items[n0:n1]:
    t = it['t']
    if it['src'] == 'G':
        a, b = it['gap']; ts = [a + (b - a) * q for q in (.2, .4, .6, .8)]
        head = f"#{it['n']} GAP {a:.2f}-{b:.2f}: missed hit?"; offs = [f'{x:.2f}' for x in ts]
    elif it['src'] == 'X':
        ts = [t - .3, t - .1, t + .1, t + .3]
        head = f"#{it['n']} ~{t:.2f} {it.get('note','')}"[:60]; offs = ['-0.3', '-0.1', '+0.1', '+0.3']
    else:
        ts = [t - .2, t, t + .2, t + .4]
        model = f"{it['src']} {it.get('hitter') or '-'}/{(it.get('side') or '-')[:1]}" + (f" {it['stroke']}" if it.get('stroke') else '')
        head = f"#{it['n']} {t:.2f} {model}"; offs = ['-0.2', '0', '+0.2', '+0.4']
    box = cbox(round(min(ts) * fps) - 6, round(max(ts) * fps) + 6)
    plan.append([(x, head if i == 0 else o) for i, (x, o) in enumerate(zip(ts, offs))] + [box])
need = {}
for r, pl in enumerate(plan):
    for c, (x, title) in enumerate(pl[:4]): need.setdefault(max(round(x * fps), 0), []).append((r, c, title, pl[4]))
cells = {}
cap.set(1, 0); f = 0; last = max(need) if need else -1
while f <= last:
    ok = cap.grab()
    if not ok: break
    if f in need:
        ok, im = cap.retrieve()
        for r, c, title, box in need[f]: cells[(r, c)] = panel(f, box, title, im.copy())
    f += 1
blank = np.zeros((270, 480, 3), np.uint8)
rows = [np.hstack([cells.get((r, c), blank) for c in range(4)]) for r in range(len(plan))]
for s in range(0, len(rows), 6):
    idx = (n0 + s) // 6
    cv2.imwrite(str(od / f'{k}{SUF}_{idx:03d}.jpg'), np.vstack(rows[s:s + 6]), [cv2.IMWRITE_JPEG_QUALITY, 75])
print(k, len(rows), 'rows', (len(rows) + 5) // 6, 'sheets')
