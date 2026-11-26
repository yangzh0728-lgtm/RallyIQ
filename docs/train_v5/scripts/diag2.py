import sys; sys.path.insert(0,'/Users/mac/Desktop/RallyIQ/train_v5/scripts')
import rt
from rallyiq.perception.shuttle.tracker import *
k,t0,t1=sys.argv[1],float(sys.argv[2]),float(sys.argv[3])
c=rt.Ctx(k);fps=c.meta.fps
class TT(ShuttleTracker):
    def _select(s,ts):
        print('--- entering select');
        for t in ts:
            if t.end>=t0*fps and t.start<=t1*fps and len(t.frames)>2: print(t.start,t.end,len(t.frames),'w',round(t.weight(s.cfg),2),'sup',t.supported,'reg',round(t.mean_region(),2),t.xy[0],t.xy[-1],'idle',s._idle(t),s._idle_shared(t))
        return super()._select(ts)
    def _target_court(s,ts):
        for t in ts:
            if t.end>=t0*fps and t.start<=t1*fps and len(t.frames)>2: print('L',t.start,t.end,len(t.frames),t.xy[0],t.xy[-1])
        return super()._target_court(ts)
T=TT(fps,c.meta.width,c.meta.height,rt.S.tracker_config({},.2),region=c.region,support=c.support,rest=c.rest,strict_support=c.strict,held=c.held)
T.track(c.cands,c.meta.frame_count)
