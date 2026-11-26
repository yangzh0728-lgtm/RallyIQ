"""Landing: last shot of each labelled rally window -> where the shuttle ends (strokes features _landing) for old vs new tracks;
count landings off our court (court coords outside the lines + 1 m) and how many differ by > 1 m."""
import sys,json; sys.path.insert(0,'/Users/mac/Desktop/RallyIQ/train_v5/scripts')
import rt,numpy as np,replay
from dataclasses import replace
import amateur_hits as AH
from rallyiq.events.config import EventsConfig
from rallyiq.events.pipeline import segment_events, off_court_views
from rallyiq.events.adapters import poses_from_contract, shuttle_from_contract, events_to_contract, calibration_segments
from rallyiq.strokes.features import shot_features
from rallyiq.strokes.classify import StrokesConfig
L=json.load(open(rt.B+'/labels/amateur_v4.json'))
from rallyiq.strokes.features import _landing
from rallyiq.perception.target_court import SegmentRegions
import functools
@functools.lru_cache(None)
def sh_pts(d): return sorted(rt.ShuttleTrack.model_validate_json(rt.rd(d,'shuttle')).points,key=lambda p:p.frame)
def fps(d): return float(json.loads(rt.rd(d,'probe'))['fps'])
def run(d,cal):
    sh=rt.ShuttleTrack.model_validate_json(rt.rd(d,'shuttle'));pl=rt.PlayerTracks.model_validate_json(rt.rd(d,'players'));pr=json.loads(rt.rd(d,'probe'))
    cfg=replace(EventsConfig(),**AH.AMATEUR_BASE); off=off_court_views(calibration_segments(cal),cfg.court_view_min_share)
    ev=events_to_contract(pr['video_id'],segment_events(shuttle_from_contract(sh),poses_from_contract(pl),cfg,video_end_ms=float(pr['duration_ms']),off_court=off),pl.match_format)
    F=shot_features(ev,sh,pl,float(pr['fps']),StrokesConfig().features); return ev,{f.shot_id:f for f in F}
tot={'old':[0,0],'new':[0,0]}
for k in L['hits']:
    cal=replay.calib(k); W=L['windows'][k]; SR=SegmentRegions(cal)
    for name,d in (('old',AH.art_dir(rt.Path(rt.H+'/train_v4/inputs'),k)),('new',f'{rt.B}/inputs_C/new_runs/{k}')):
        ev,F=run(str(d),cal); n=off=0
        for r in ev.rallies:
            shots=[s for s in ev.shots if s.rally_id==r.rally_id]
            if not shots or not any(a<=shots[-1].t_ms/1000<=b+1 for a,b in W): continue
            s0=shots[-1]; pts=[p for p in sh_pts(d) if s0.frame<p.frame<=min(r.end_frame,s0.frame+int(2.5*fps(d)))]
            if len(pts)<3: continue
            lp=_landing(pts,100.0); reg=SR.for_frame(lp.frame)
            if reg is None: continue
            n+=1; off+=reg.floor_outside_m([lp.x,lp.y])>1.0
        tot[name][0]+=n;tot[name][1]+=off
    print(k,tot,flush=True)
