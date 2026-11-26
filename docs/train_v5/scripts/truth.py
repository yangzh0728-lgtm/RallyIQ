"""Pseudo truth for shuttle positions from labelled hits.
Between consecutive labelled hits (0.25-2.5 s apart, opposite sides) pick the candidate tracklet chain that leaves the
hitter's wrist and reaches the next hitter's wrist. -> inputs/flights.json [{key,f0,f1,pts:{frame:[x,y]}}]
Independent of what the tracker selected (uses raw candidates with no court lock)."""
import sys; sys.path.insert(0,'/Users/mac/Desktop/RallyIQ/train_v5/scripts')
import rt,json,math,numpy as np
from pathlib import Path
from multiprocessing import Pool
import amateur_hits as AH
from rallyiq.perception.shuttle.tracker import ShuttleTracker
B4=Path(rt.H+'/train_v4')
TR=AH.load_truth(B4/'inputs',B4/'labels/amateur_v3.json')
def hitter(c,f,lab,w=3):
    """players (h, wrists) who may be the hitter around frame f"""
    for d in (0,-1,1,-2,2,-3,3):
        fr=c.byf.get(f+d)
        if not fr: continue
        ps=[p for p in fr if (p.player_id==lab['hitter'] if lab.get('hitter') else p.side==lab['side'])]
        if not ps and lab.get('hitter'): ps=[p for p in fr if p.side==lab['side']]
        if ps: return [(p.bbox[3]-p.bbox[1],[(k[0],k[1]) for i,k in enumerate(p.keypoints) if i in(9,10) and k[2]>.2] or [((p.bbox[0]+p.bbox[2])/2,p.bbox[1])]) for p in ps]
    return []
def wd(ps,x,y): return min((min(math.dist(w,(x,y)) for w in ws)/max(h,60) for h,ws in ps),default=9)
def build(k):
    c=rt.Ctx(k); fps=c.meta.fps
    c.byf={f.frame:f.players for f in c.pl.frames}
    labs=sorted([l for l in TR[k]['labels']],key=lambda l:l['t'])
    T=ShuttleTracker(fps,c.meta.width,c.meta.height,rt.S.tracker_config({},.2))
    out=[];tot=0
    for a,b in zip(labs,labs[1:]):
        if a['kind']!='pos' or b['kind']!='pos' or a['side']==b['side'] or not .25<b['t']-a['t']<2.6: continue
        tot+=1
        f0,f1=round(a['t']*fps),round(b['t']*fps); pad=round(.12*fps)
        pa,pb=hitter(c,f0,a),hitter(c,f1,b)
        if not pa or not pb: continue
        sub={f:c.cands.get(f,[]) for f in range(f0-pad,f1+pad+1)}
        ts=[t for t in T._link(sub,{f:list(range(len(v))) for f,v in sub.items()}) if len(t.frames)>=4]
        # cut at turns so a tracklet that links hit to hit splits; keep pieces
        ps=[]
        for t in ts: 
            # trim to the interval
            idx=[i for i,f in enumerate(t.frames) if f0-pad/2<=f<=f1+pad/2]
            if len(idx)>=3: ps.append(t.subset(idx))
        # chain search: start piece near A, end piece near B, pieces time ordered; minimise cost
        best=None
        def ext(chain):
            nonlocal best
            t=chain[-1]
            de=wd(pb,*t.xy[-1]); late=(f1-t.end)/fps
            n=sum(len(x.frames) for x in chain); cov=n/(f1-f0+1)
            if de<.75 and late<.2 and cov>.5:
                s=cov-.15*(len(chain)-1)-.2*de
                if best is None or s>best[0]: best=(s,list(chain))
            if len(chain)>=3: return
            v=t.velocity() or (0,0)
            for u in ps:
                dt=u.start-t.end
                if 0<dt<=.5*fps:
                    ex,ey=t.xy[-1][0]+v[0]*dt,t.xy[-1][1]+v[1]*dt
                    if math.dist((ex,ey),u.xy[0])<=60+.6*math.hypot(*v)*dt: ext(chain+[u])
        for t in ps:
            if wd(pa,*t.xy[0])<.75 and (t.start-f0)/fps<.2: ext([t])
        if best:
            pts={f:[round(x,1),round(y,1)] for t in best[1] for f,(x,y) in zip(t.frames,t.xy) if f0<=f<=f1}
            out.append(dict(key=k,f0=f0,f1=f1,t=a['t'],pts=pts,s=round(best[0],2)))
    return k,tot,out
if __name__=='__main__':
    al=[]
    with Pool(6) as p:
        for k,tot,o in p.imap(build,rt.IDS): al+=o; print(k,tot,len(o),sum(len(x['pts']) for x in o),flush=True)
    json.dump(al,open(rt.B+'/inputs/flights.json','w'))
