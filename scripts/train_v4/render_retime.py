"""Re-timing sheets for the 58 unrefined qa6 hits: overview with the shuttle path (+-1 s, labelled every 0.2 s,
rule candidates circled) + 16 hitter-side crops at -0.8..+0.7 s (step 0.1). -> review/qa6_retime/NNN.jpg"""
import cv2, json, gzip, os, sys, pickle, numpy as np
sys.path.insert(0, os.path.expanduser('~/Desktop/RallyIQ/train_v4/repo')); sys.path.insert(0, os.path.expanduser('~/Desktop/RallyIQ/train_v4/repo/scripts'))
B3 = os.path.expanduser('~/Desktop/RallyIQ/train_v3'); B = os.path.expanduser('~/Desktop/RallyIQ/train_v4')
I = f'{B3}/inputs'
clips = json.load(open(f'{I}/videos/index.json'))
runs = {'own_20260529_0006': 'vid_97a8ee07f5fe', 'own_20260529_0003': 'vid_f52af81fca49', 'own_20260529_0004': 'vid_355e0f0cc3a6',
        'own_20260529_0008': 'vid_b4eb017d39e6', 'own_20260906_0615': 'vid_efbc5a6de4af', 'own_20260906_0620': 'vid_df0e89955052'}
v2 = json.load(open(f'{I}/hits_truth_v2.json'))['hits']
todo = [h for h in v2 if h['sure'] and h['in_play'] is True and h['align_src'] == 'unrefined']
cands = {c.cid: c.t for c in pickle.load(open(f'{B3}/ckpt/cands2/am_msc0.3.pkl', 'rb'))}
out = f'{B}/review/qa6_retime'; os.makedirs(out, exist_ok=True)
STEPS = [round(-0.8 + 0.1 * k, 1) for k in range(16)]
def rd(p): return json.load(gzip.open(p + '.gz')) if not os.path.exists(p) else json.load(open(p))
def crop(im, box, ar, h):
    x0, y0, x1, y1 = box; cx, cy = (x0 + x1) / 2, (y0 + y1) / 2; bh = max(y1 - y0, (x1 - x0) / ar); bw = bh * ar
    x0, x1, y0, y1 = int(cx - bw / 2), int(cx + bw / 2), int(cy - bh / 2), int(cy + bh / 2)
    P = 3000; pad = cv2.copyMakeBorder(im, P, P, P, P, cv2.BORDER_CONSTANT)
    return cv2.resize(pad[y0 + P:y1 + P, x0 + P:x1 + P], (int(h * ar), h))
meta = []
for clip in sorted({h['clip'] for h in todo}):
    art = f"{I}/own_v12/{runs[clip]}/artifacts"
    fps = rd(f'{art}/probe.json')['fps']
    pl = {f['frame']: f['players'] for f in rd(f'{art}/players.json')['frames']}
    sh = {p['frame']: p for p in rd(f'{art}/shuttle.json')['points']}
    caps = {}
    def frame(t):
        c = next((c for c in clips[clip] if c['t0'] <= t <= c['t1']), None)
        if c is None: return None
        cap = caps.get(c['file']) or cv2.VideoCapture(f"{I}/videos/{c['file']}"); caps[c['file']] = cap
        cap.set(cv2.CAP_PROP_POS_FRAMES, round((t - c['t0']) * cap.get(cv2.CAP_PROP_FPS))); ok, im = cap.read()
        return im if ok else None
    def players(f):
        for d in (0, -1, 1, -2, 2, -3, 3):
            if f + d in pl: return pl[f + d]
        return []
    for h in sorted([h for h in todo if h['clip'] == clip], key=lambda h: h['t']):
        t = h['t']; f0 = round(t * fps)
        ov = frame(t); ov = np.zeros((1080, 1920, 3), np.uint8) if ov is None else ov.copy()
        for f in range(f0 - int(fps), f0 + int(fps) + 1):
            p = sh.get(f)
            if not p or p.get('x') is None: continue
            dt = (f - f0) / fps; col = (255, 80, 0) if dt < 0 else (0, 80, 255)
            cv2.circle(ov, (int(p['x']), int(p['y'])), 4, col, -1)
            if abs(dt * 5 - round(dt * 5)) < 0.5 / fps * 5:
                cv2.putText(ov, f'{dt:+.1f}', (int(p['x']) + 6, int(p['y']) - 6), 0, 0.8, (255, 255, 255), 2)
        ct = [x for x in cands[clip] if abs(x - t) <= 1.0]
        for x in ct:
            p = sh.get(round(x * fps))
            if p and p.get('x') is not None:
                cv2.circle(ov, (int(p['x']), int(p['y'])), 16, (0, 255, 255), 3)
        ov = cv2.resize(ov[330:1080, 160:1760], (1200, 562))
        side = [p for p in players(f0) if p.get('side') == h['side']] or players(f0)
        if side:
            bx = np.array([p['bbox'] for p in side]); box = [bx[:, 0].min(), bx[:, 1].min(), bx[:, 2].max(), bx[:, 3].max()]
        else: box = [700, 300, 1200, 800]
        w, hh = box[2] - box[0], box[3] - box[1]; box = [box[0] - 0.5 * w, box[1] - 0.5 * hh, box[2] + 0.5 * w, box[3] + 0.05 * hh]
        parts = []
        for dt in STEPS:
            im = frame(t + dt); im = np.zeros((1080, 1920, 3), np.uint8) if im is None else im.copy()
            sp = sh.get(round((t + dt) * fps))
            if sp and sp.get('x') is not None: cv2.circle(im, (int(sp['x']), int(sp['y'])), 12, (0, 255, 0), 2)
            c = crop(im, box, 0.75, 300)
            cv2.putText(c, f'{dt:+.1f}', (4, 290), 0, 0.8, (0, 255, 255) if dt == 0 else (255, 255, 255), 2); parts.append(c)
        r1 = cv2.hconcat(parts[:8]); r2 = cv2.hconcat(parts[8:])
        W = r1.shape[1]; ovr = np.zeros((562, W, 3), np.uint8); ovr[:, (W - 1200) // 2:(W - 1200) // 2 + 1200] = ov
        nb = [f"{x['t'] - t:+.2f}{x['side'][0]}{'' if x['sure'] else '?'}" for x in v2 if x['clip'] == clip and 0 < abs(x['t'] - t) <= 2.0]
        hdr = np.zeros((40, W, 3), np.uint8)
        txt = f"#{h['id']} {clip} t={t:.2f} side={h['side']} | cands {' '.join(f'{x - t:+.2f}' for x in ct)} | other labels {' '.join(nb)}"
        cv2.putText(hdr, txt, (6, 28), 0, 0.75, (0, 255, 255), 2)
        cv2.imwrite(f"{out}/{h['id']:03d}.jpg", cv2.vconcat([hdr, ovr, r1, r2]), [cv2.IMWRITE_JPEG_QUALITY, 80])
        meta.append({'id': h['id'], 'clip': clip, 't': t, 'side': h['side'], 'cands': [round(x - t, 3) for x in ct], 'others': nb})
json.dump(meta, open(f'{out}/index.json', 'w'), indent=0)
print(len(meta))
