"""Amateur stroke rows for the court model: per labelled hit (amateur_v3, stroke typed) matched to an events shot
(events re-run with the bundled v3b scorer + own-footage settings), the court feature row, the video-model
probabilities, a pose 'wrist above shoulder' feature and the label. -> ckpt/stroke_rows.pkl"""
import sys, json, math, pickle, gzip
from dataclasses import replace
from pathlib import Path
import numpy as np
import os; B = Path('~/Desktop/RallyIQ/train_v5').expanduser(); B4 = Path('~/Desktop/RallyIQ/train_v4').expanduser(); sys.path[:0] = [str(B / 'repo'), str(B / 'repo/scripts')]
import amateur_hits as AH
from rallyiq.contracts import EventsResult, PlayerTracks, ShuttleTrack, CourtCalibration
from rallyiq.events.adapters import calibration_segments, events_to_contract, poses_from_contract, shuttle_from_contract
from rallyiq.events.config import EventsConfig
from rallyiq.events.pipeline import off_court_views, segment_events
from rallyiq.strokes import court as CM
from rallyiq.strokes.classify import classify
from rallyiq.strokes.features import shot_features
from rallyiq.strokes.detail import get_model as get_attrs
from rallyiq.strokes.learned import BUNDLED, make_model_fn as video_model
from rallyiq.strokes.classify import StrokesConfig
SS = len(sys.argv) > 1 and sys.argv[1] == 'ss'
if SS:
    import csv
    from rallyiq.strokes.taxonomy import from_shuttleset
    T = Path(os.environ.get('SSDIR', '~/Desktop/RallyIQ/train_v3/inputs/public/shuttleset_test_v3')).expanduser()
    idx = json.load(open(T / 'index.json'))
    labels = {'hits': {}}; ART = {}
    for cid, m in idx.items():
        lab = json.loads((T / 'labels' / f'{cid}.json').read_text(encoding='latin-1'))
        R = list(csv.DictReader(open(B4 / 'inputs/shuttleset_set' / m['video'] / f"{cid.split('_')[1]}.csv", encoding='utf-8')))
        st = [(float(r['frame_num']) * 1000 / m['fps'], from_shuttleset((r.get('type') or '').strip())[0]) for r in R if r.get('frame_num')]
        hs = []
        for h in lab['hits']:
            c = min(st, key=lambda x: abs(x[0] - h['t_ms']))
            if abs(c[0] - h['t_ms']) <= 60 and c[1]:
                hs.append({'t': (h['t_ms'] - lab['range_ms'][0]) / 1000, 'stroke': c[1], 'stroke_sure': True, 'status': 'reviewed_sure', 'side': h['side']})
        labels['hits'][cid] = hs; ART[cid] = Path(m['artifacts'])
else:
    labels = json.load(open(B / os.environ.get('LABELS', 'labels/amateur_v4_newids.json')))
court = CM.get_model(); attrs = get_attrs(); vid = video_model(BUNDLED)
cfg = EventsConfig() if SS else replace(EventsConfig(), **AH.AMATEUR_BASE)
rows = []
for key, hs in labels['hits'].items():
    lab = [h for h in hs if h['status'] in ('reviewed_sure', 'reviewed_unsure') and h.get('stroke')]
    if not lab: continue
    d = ART[key] if SS else AH.art_dir(B / os.environ.get('INP', 'inputs_C'), key)
    probe = json.loads(AH.read_json(d, 'probe')); fps = float(probe['fps'])
    sh = ShuttleTrack.model_validate_json(AH.read_json(d, 'shuttle')); pl = PlayerTracks.model_validate_json(AH.read_json(d, 'players'))
    cj = AH.read_json(d, 'calibration'); cal = CourtCalibration.model_validate_json(cj) if cj else None
    off = off_court_views(calibration_segments(cal), cfg.court_view_min_share) if cfg.mask_off_court_views else []
    rallies = segment_events(shuttle_from_contract(sh), poses_from_contract(pl), cfg, video_end_ms=float(probe['duration_ms']), off_court=off)
    ev = events_to_contract(probe['video_id'], rallies, pl.match_format)
    feats = shot_features(ev, sh, pl, fps, StrokesConfig().features)
    high = CM.contact_high(ev, sh, pl, feats, fps, attrs)
    ids, X, rids = CM.video_features(ev, high, court.depth_q)
    pv = vid(ev, sh, pl, feats) if vid else {}
    V = np.asarray([[pv.get(i, {}).get(c, math.nan) for c in court.classes] if i in pv else [math.nan] * len(court.classes) for i in ids])
    # pose: wrist above shoulder (image y) of the hitter at the contact frame (+-2 frames), in torso lengths
    pf = {f.frame: f for f in pl.frames}
    shots = {s.shot_id: s for s in ev.shots}
    def overhead(s):
        best = math.nan
        for df in (0, -1, 1, -2, 2, -3, 3):
            fr = pf.get(s.frame + df)
            if fr is None: continue
            for p in fr.players:
                if p.player_id != s.hitter or not p.keypoints: continue
                k = p.keypoints
                def kp(i):
                    v = k[i]; return (v[0], v[1]) if (len(v) < 3 or v[2] >= 0.3) else None
                sh_ = [kp(5), kp(6)]; wr = [kp(9), kp(10)]; hp = [kp(11), kp(12)]
                sh_ = [x for x in sh_ if x]; wr = [x for x in wr if x]; hp = [x for x in hp if x]
                if not sh_ or not wr: continue
                sy = min(y for _, y in sh_); wy = min(y for _, y in wr)
                torso = (np.mean([y for _, y in hp]) - np.mean([y for _, y in sh_])) if hp else (p.bbox[3] - p.bbox[1]) / 3
                if torso <= 1: continue
                return (sy - wy) / torso
        return best
    spts = {q.frame: q for q in sh.points}
    order = sorted(ev.shots, key=lambda s: s.frame)
    nxt = {a.shot_id: (b.frame if b.rally_id == a.rally_id else None) for a, b in zip(order, order[1:])}
    def bbox_h(s):
        for df in (0, -1, 1, -2, 2):
            fr = pf.get(s.frame + df)
            if fr:
                for p in fr.players:
                    if p.player_id == s.hitter: return p.bbox[3] - p.bbox[1]
        return 200.0
    def rise(s):
        y0 = next((spts[s.frame + d].y for d in (0, 1, -1, 2, -2, 3, -3) if s.frame + d in spts), None)
        if y0 is None: return math.nan, math.nan
        end = nxt.get(s.shot_id) or s.frame + int(1.5 * fps)
        end = min(end, s.frame + int(2.0 * fps))
        ys = [spts[f].y for f in range(s.frame + 1, end) if f in spts]
        if len(ys) < 4: return math.nan, math.nan
        h = max(bbox_h(s), 30.0)
        return (y0 - min(ys)) / h, (ys[-1] - y0) / h
    t = {i: shots[i].t_ms / 1000 for i in ids}
    used = set(); n = 0
    for h in sorted(lab, key=lambda h: h['t']):
        c = [j for j, i in enumerate(ids) if j not in used and abs(t[i] - h['t']) <= 0.3]
        if not c: continue
        j = min(c, key=lambda j: abs(t[ids[j]] - h['t'])); used.add(j); n += 1
        s = shots[ids[j]]
        rows.append({'key': key, 't': h['t'], 'y': h['stroke'], 'sure': bool(h.get('stroke_sure')), 'X': X[j].tolist(), 'V': V[j].tolist(),
                     'oh': overhead(s), 'rise': rise(s)[0], 'fall': rise(s)[1], 'r4': str(h.get('review', '')).startswith('claude_v4_r4'), 'side': h.get('side')})
    print(key, len(lab), 'typed labels,', n, 'matched', flush=True)
pickle.dump({'rows': rows, 'classes': court.classes}, open(B / (os.environ.get('ROWS_SS', 'ckpt/stroke_rows_ss.pkl') if SS else os.environ.get('ROWS', 'ckpt/stroke_rows.pkl')), 'wb'))
print(len(rows), 'rows,', sum(r['sure'] for r in rows), 'trusted')
