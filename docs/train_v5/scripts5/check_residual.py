"""Plumbing check: classify() with the bundled residual on the amateur videos == the training-row computation."""
import sys, json, math, pickle
from dataclasses import replace
from pathlib import Path
import numpy as np
import os; B = Path('~/Desktop/RallyIQ/train_v5').expanduser(); B4 = Path('~/Desktop/RallyIQ/train_v4').expanduser(); sys.path[:0] = [str(B / 'repo'), str(B / 'repo/scripts')]
import amateur_hits as AH
from rallyiq.contracts import PlayerTracks, ShuttleTrack, CourtCalibration
from rallyiq.events.adapters import calibration_segments, events_to_contract, poses_from_contract, shuttle_from_contract
from rallyiq.events.config import EventsConfig
from rallyiq.events.pipeline import off_court_views, segment_events
from rallyiq.strokes import court as CM
from rallyiq.strokes.classify import classify
from rallyiq.strokes.detail import get_model as get_attrs
from rallyiq.strokes.learned import BUNDLED, make_model_fn as video_model
rows = pickle.load(open(B / os.environ.get('ROWS', 'ckpt/stroke_rows.pkl'), 'rb'))['rows']
cfg = replace(EventsConfig(), **AH.AMATEUR_BASE); court = CM.get_model(); attrs = get_attrs(); vid = video_model(BUNDLED); res = CM.get_residual()
tot = {'base': [0, 0], 'res': [0, 0]}; dif = []
for key in sorted({r['key'] for r in rows})[:4]:
    d = AH.art_dir(B / os.environ.get('INP', 'inputs_C'), key); probe = json.loads(AH.read_json(d, 'probe')); fps = float(probe['fps'])
    sh = ShuttleTrack.model_validate_json(AH.read_json(d, 'shuttle')); pl = PlayerTracks.model_validate_json(AH.read_json(d, 'players'))
    cj = AH.read_json(d, 'calibration'); cal = CourtCalibration.model_validate_json(cj) if cj else None
    off = off_court_views(calibration_segments(cal), cfg.court_view_min_share)
    ev = events_to_contract(probe['video_id'], segment_events(shuttle_from_contract(sh), poses_from_contract(pl), cfg, video_end_ms=float(probe['duration_ms']), off_court=off), pl.match_format)
    own = CM.own_shot_features(ev, sh, pl, fps)
    t = {s.shot_id: s.t_ms / 1000 for s in ev.shots}
    for name, r_ in (('base', None), ('res', res)):
        out = classify(ev, sh, pl, fps, model=CM.make_model_fn(vid, court, attrs, fps, seq=True, residual=r_), model_name=name, attrs=attrs)
        pred = {s.shot_id: s.stroke.shot_type for s in out.shots}
        for r in rows:
            if r['key'] != key or not r['sure']: continue
            i = min(t, key=lambda i: abs(t[i] - r['t']))
            if abs(t[i] - r['t']) > 0.3: continue
            tot[name][0] += pred[i] == r['y']; tot[name][1] += 1
            if name == 'res': dif.append(abs((own[i]['rise'] if not math.isnan(own[i]['rise']) else 0) - (r['rise'] if not math.isnan(r['rise']) else 0)))
print(tot, 'max |rise diff|', max(dif) if dif else None)
