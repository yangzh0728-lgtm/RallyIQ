import sys,json; sys.path.insert(0,'/Users/mac/Desktop/RallyIQ/train_v5/scripts')
import rt
from dataclasses import replace
import amateur_hits as AH
from rallyiq.events.config import EventsConfig
from rallyiq.events.pipeline import segment_events
from rallyiq.events.adapters import poses_from_contract, shuttle_from_contract
k,inp=sys.argv[1],sys.argv[2]; ts=[float(x) for x in sys.argv[3:]]
d=f'{rt.B}/{inp}/new_runs/{k}' if inp!='old' else rt.art(k)
sh=rt.ShuttleTrack.model_validate_json(rt.rd(d,'shuttle'));pl=rt.PlayerTracks.model_validate_json(rt.rd(d,'players'))
cfg=replace(EventsConfig(),**AH.AMATEUR_BASE)
col=[];R=segment_events(shuttle_from_contract(sh),poses_from_contract(pl),cfg,collect=col)
for rid,x,c,pt in col:
    t=pt.t_ms/1000
    if any(abs(t-q)<.3 for q in ts):
        print(inp,rid,round(t,2),'p',round(getattr(c,'p',0) or 0,2),'conf',round(c.conf,2) if hasattr(c,'conf') else '', c.reasons, c.best.pose.player_id if c.best else None, c.best.pose.team if c.best else None,[ (p.pose.player_id,round(p.score,2)) for p in c.prox][:4])
from rallyiq.events.learned import ALL_FEATURES
import numpy as np
X={}
for rid,x,c,pt in col:
    t=pt.t_ms/1000
    if any(abs(t-q)<.3 for q in ts): print(round(t,2),{f:round(float(v),2) for f,v in zip(ALL_FEATURES,x)})
for r in R:
    for h in getattr(r,'hits',[]) or getattr(r,'shots',[]):
        pass
print([ (r.rally_id if hasattr(r,'rally_id') else '', round(r.start_ms/1000,2), round(r.end_ms/1000,2)) for r in R if any(r.start_ms/1000-2<q<r.end_ms/1000+2 for q in ts)] if R and hasattr(R[0],'start_ms') else type(R[0]))
