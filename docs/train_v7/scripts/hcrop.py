"""hcrop.py KEY...: for every labelled positive hit in doubles videos, a crop of the hit frame with all raw pose
detections (score>=0.2, pose dump) numbered, the v5 track id of each, and the v5 shuttle point. Writes
review/hl/<key>_<n>.jpg mosaics (6 per sheet) and review/hl/<key>.json with the candidate boxes per hit."""
import sys, os, json, cv2, numpy as np
sys.path.insert(0, '/Users/mac/Desktop/RallyIQ/train_v5/scripts')
import rt
B6 = os.path.expanduser('~/Desktop/RallyIQ/train_v6'); OUT = B6 + '/review/hl'; os.makedirs(OUT, exist_ok=True)
LAB = json.load(open(rt.B + '/labels/amateur_v4_newids.json'))
def iou(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0]); h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0: return 0
    i = w * h; return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)
def run(k, players=None, tag=''):
    fps = float(rt.IDS[k][2]); d = np.load(f'{rt.B}/dumps/{k}.npz'); fr = d['frame']; bb = d['bbox']; sc = d['score']
    P = json.load(open(players or f'{rt.B}/players_v5/{k}.json')); PB = {f['frame']: f['players'] for f in P['frames']}
    S = json.load(open(f'{rt.B}/inputs_C/new_runs/{k}/shuttle.json')); SP = {p['frame']: p for p in S['points']}
    src = f'{rt.H}/train_v4/inputs/sources/{k}.mp4'; cap = cv2.VideoCapture(src)
    hits = [h for h in LAB['hits'][k] if h['status'] in ('reviewed_sure', 'reviewed_unsure')]
    tiles, meta = [], []
    for n, h in enumerate(sorted(hits, key=lambda h: h['t'])):
        f = round(h['t'] * fps); f2 = f - f % 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, f2); ok, img = cap.read()
        if not ok: continue
        ix = np.where(fr == f2)[0]; dets = [(bb[i].tolist(), float(sc[i])) for i in ix]
        trk = [p for p in PB.get(f2, []) if not p.get('interpolated')]
        cand = []
        for j, (b, s) in enumerate(dets):
            m = max(trk, key=lambda p: iou(p['bbox'], b), default=None)
            pid = m['player_id'] if m is not None and iou(m['bbox'], b) > .5 else None
            cand.append({'i': j, 'bbox': [round(v, 1) for v in b], 'score': round(s, 3), 'pid': pid})
        # crop: the labelled side's half of the image band around detections
        ys = [c['bbox'][3] for c in cand]; xs = [c['bbox'][0] for c in cand] + [c['bbox'][2] for c in cand]
        if h['side'] == 'far': y0, y1 = 300, 760
        else: y0, y1 = 520, 1080
        x0, x1 = 0, 1920
        im = img.copy()
        for c in cand:
            b = [int(v) for v in c['bbox']]; col = (0, 200, 0) if c['pid'] else (0, 0, 255)
            cv2.rectangle(im, (b[0], b[1]), (b[2], b[3]), col, 2)
            cv2.putText(im, f"{c['i']}{(':'+c['pid']) if c['pid'] else ''} .{int(c['score']*100)}", (b[0], max(b[1] - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, .8, col, 2)
        for g in range(f - 3, f + 4):
            if g in SP: cv2.circle(im, (int(SP[g]['x']), int(SP[g]['y'])), 7 if g == f else 3, (0, 255, 255), 2)
        crop = im[y0:y1, x0:x1]; crop = cv2.resize(crop, (1280, int(1280 * (y1 - y0) / 1920)))
        cv2.putText(crop, f"#{n} t={h['t']:.2f} side={h['side']} lab={h.get('hitter')} st={h['status'][9:]}", (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        cv2.putText(crop, (h.get('note') or '')[:90], (8, 62), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 0), 2)
        tiles.append(crop); meta.append({'n': n, 't': h['t'], 'frame': f2, 'side': h['side'], 'label': h.get('hitter'), 'status': h['status'], 'cand': cand})
    json.dump(meta, open(f'{OUT}/{k}{tag}.json', 'w'))
    for s in range(0, len(tiles), 4):
        grp = tiles[s:s + 4]; W = 1280; Hh = max(t.shape[0] for t in grp)
        grp = [cv2.copyMakeBorder(t, 0, Hh - t.shape[0] + 6, 0, 0, cv2.BORDER_CONSTANT) for t in grp]
        cv2.imwrite(f'{OUT}/{k}{tag}_{s // 4:02d}.jpg', np.vstack(grp), [cv2.IMWRITE_JPEG_QUALITY, 80])
    print(k, len(tiles))
if __name__ == '__main__':
    for k in sys.argv[1:]: run(k)
