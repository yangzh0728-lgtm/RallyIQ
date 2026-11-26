import sys; sys.path.insert(0,'/Users/mac/Desktop/RallyIQ/train_v5/scripts')
import rt
from rallyiq.perception.shuttle.tracker import *
k,t0,t1=sys.argv[1],float(sys.argv[2]),float(sys.argv[3]);D=sys.argv[4] if len(sys.argv)>4 else 'inputs_B'
c=rt.Ctx(k,f'{rt.B}/{D}/new_runs/{k}',f'{rt.B}/inputs/cands/{k}.json');fps=c.meta.fps;o=c.own
class TT(ShuttleTracker):
    def _ownership(s,ts):
        for t in ts:
            if t.end>=t0*fps and t.start<=t1*fps and len(t.frames)>2:
                h=s._hard_turns(t);z=o.logit(t.frames,t.xy,h,s.concurrency);print(t.start,t.end,len(t.frames),t.xy[0],t.xy[-1],{a:round(b,2) for a,b in (o.features(t.frames,t.xy,h) or {}).items()},[(t.frames[i],t.xy[i]) for i in h],'z',z)
        return super()._ownership(ts)
T=TT(fps,c.meta.width,c.meta.height,rt.S.tracker_config({},.2),region=c.region,support=c.support,rest=c.rest,strict_support=c.strict,held=c.held,foreign=o.logit,contact=o.contact)
T.track(c.cands,c.meta.frame_count)
