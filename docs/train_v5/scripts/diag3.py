import sys; sys.path.insert(0,'/Users/mac/Desktop/RallyIQ/train_v5/scripts')
import rt
from rallyiq.perception.shuttle.tracker import *
k,t0,t1=sys.argv[1],float(sys.argv[2]),float(sys.argv[3])
c=rt.Ctx(k);fps=c.meta.fps;o=rt.S.CourtOwnership(c.pl,fps,c.meta.width)
class TT(ShuttleTracker):
    def _select(s,ts):
        print('prior',s.foreign_prior)
        for t in ts:
            if t.end>=t0*fps and t.start<=t1*fps and len(t.frames)>2: print('S',t.start,t.end,len(t.frames),'w',round(t.weight(s.cfg),2),'ours',round(t.ours,3),t.xy[0],t.xy[-1])
        return super()._select(ts)
    def _ownership(s,ts):
        for t in ts:
            if t.end>=t0*fps and t.start<=t1*fps and len(t.frames)>2:
                print('L',t.start,t.end,len(t.frames),t.xy[0],t.xy[-1],{a:round(b,2) for a,b in (o.features(t.frames,t.xy,s._turns(t)) or {}).items()},[t.frames[i] for i in s._turns(t)])
        return super()._ownership(ts)
T=TT(fps,c.meta.width,c.meta.height,rt.S.tracker_config({},.2),region=c.region,support=c.support,rest=c.rest,strict_support=c.strict,held=c.held,foreign=c.foreign)
T.track(c.cands,c.meta.frame_count)
