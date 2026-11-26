import sys; sys.path.insert(0,'/Users/mac/Desktop/RallyIQ/train_v5/scripts')
import rt,json,numpy as np
from multiprocessing import Pool
from collections import Counter
from rallyiq.perception.shuttle.tracker import ShuttleTracker
def f(a):
    n,c=a
    c=rt.Ctx(*c); sx=c.meta.width/c.size[0]; cap={}
    class TT(ShuttleTracker):
        def _target_court(s,ts): cap['t']=ts; return super()._target_court(ts)
    T=TT(c.meta.fps,c.meta.width,c.meta.height,rt.S.tracker_config({},.2)); T.track({f:[(x*sx,y*sx,s) for x,y,s in v] for f,v in c.cands.items()},c.meta.frame_count)
    out=[]
    for ml in (.25,.4,.6):
        cnt=Counter()
        for t in cap['t']:
            if np.mean(t.scores)<.5 or (t.end-t.start)<ml*c.meta.fps or T._path_speed(t)<.1 or t.weight(T.cfg)<=0: continue
            ext=max(np.ptp([p[0] for p in t.xy]),np.ptp([p[1] for p in t.xy]))
            if ext<.06*c.meta.width: continue
            for g in range(t.start,t.end+1): cnt[g]+=1
        out.append(round(sum(1 for v in cnt.values() if v>1)/max(len(cnt),1),3))
    return n,out
if __name__=='__main__':
    idx=json.load(open(rt.H+'/train_v3/inputs/public/shuttleset_test_v3/index.json'));it=json.load(open(rt.H+'/train_v3/inputs/public/shuttleset_train_v3/index.json'))
    J=[(k,(k,)) for k in rt.IDS]+[(k,('x',e['artifacts'],f"{rt.B}/inputs/ss_cands/{s}/{e['artifacts'].split('/')[-2]}.json")) for s,i in (('shuttleset_test_v3',idx),('shuttleset_train_v3',it)) for k,e in i.items() if 'artifacts' in e]
    with Pool(5) as p: R=p.map(f,J)
    [print(*r) for r in R[:31]];json.dump(dict((r[0],r[1][1]) for r in R),open(rt.B+'/inputs/conc.json','w'))

