import sys; sys.path.insert(0,'/Users/mac/Desktop/RallyIQ/train_v5/scripts')
import rt,json,numpy as np
from rallyiq.perception.shuttle.tracker import ShuttleTracker
idx=json.load(open(rt.H+'/train_v3/inputs/public/shuttleset_test_v3/index.json'))
for n,e in [(k,None) for k in ('own_20260906_0618','own_20260302_07','own_20260324_0241')]+[(k,idx[k]) for k in ('sx01_set2','sx01_set3','sx29_set1')]:
    c=rt.Ctx(n) if e is None else rt.Ctx('x',e['artifacts'],f"{rt.B}/inputs/ss_cands/shuttleset_test_v3/{e['artifacts'].split('/')[-2]}.json");cap={};sx=c.meta.width/c.size[0]
    class TT(ShuttleTracker):
        def _target_court(s,ts): cap['t']=ts; return super()._target_court(ts)
    T=TT(c.meta.fps,c.meta.width,c.meta.height,rt.S.tracker_config({},.2)); T.track({f:[(x*sx,y*sx,s) for x,y,s in v] for f,v in c.cands.items()},c.meta.frame_count)
    ts=[t for t in cap['t'] if t.end-t.start>=.4*c.meta.fps and T.diag and t.weight(T.cfg)>0 and not T._idle(t) and max(np.ptp([p[0] for p in t.xy]),np.ptp([p[1] for p in t.xy]))>.06*c.meta.width]
    def conc(ts,mo):
        tot=set();two=set()
        for t in ts: tot|=set(range(t.start,t.end+1))
        for i,a in enumerate(ts):
            for b in ts[i+1:]:
                lo,hi=max(a.start,b.start),min(a.end,b.end)
                if hi-lo>=mo*c.meta.fps: two|=set(range(lo,hi+1))
        return round(len(two)/max(len(tot),1),3)
    print(n,conc(ts,0),conc(ts,.4),conc([t for t in ts if np.mean(t.scores)>.5],.4),conc([t for t in ts if np.mean(t.scores)>.5],0),c.meta.frame_count)
    if e and n=='sx01_set2':
        for t in ts: print('  ',t.start,t.end,len(t.frames),round(np.mean(t.scores),2),t.xy[0],t.xy[-1])
