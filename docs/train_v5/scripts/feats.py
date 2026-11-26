"""Selected (kept) tracklet pieces in the labelled windows with contact features."""
import sys; sys.path.insert(0,'/Users/mac/Desktop/RallyIQ/train_v5/scripts')
import rt,json,math,numpy as np,pickle
from pathlib import Path
from multiprocessing import Pool
import amateur_hits as AH
from rallyiq.perception.shuttle.tracker import ShuttleTracker
B4=Path(rt.H+'/train_v4')
TR=AH.load_truth(B4/'inputs',B4/'labels/amateur_v3.json')
def wdist(c,f,x,y,ext=2):
    best=9
    for d in range(-ext,ext+1):
        for p in c.byf.get(f+d,()):
            h=max(p.bbox[3]-p.bbox[1],70)
            ws=[k for i,k in enumerate(p.keypoints) if i in(9,10) and k[2]>.2]
            pts=[(k[0],k[1]) for k in ws] or [((p.bbox[0]+p.bbox[2])/2,p.bbox[1]+.2*h)]
            best=min(best,min(math.dist(q,(x,y)) for q in pts)/h)
    return best
def feat(c,T,t):
    fps=c.meta.fps;n=len(t.frames);k=max(2,round(.1*fps));ext=round(.2*fps)
    def end(idx,sign):
        i=idx[0];j=idx[-1]
        d=wdist(c,t.frames[i],*t.xy[i]);
        df=abs(t.frames[j]-t.frames[i]) or 1
        vx,vy=(t.xy[i][0]-t.xy[j][0])/df,(t.xy[i][1]-t.xy[j][1])/df
        for s in range(1,ext+1):
            d=min(d,wdist(c,t.frames[i]+sign*s,t.xy[i][0]+vx*s,t.xy[i][1]+vy*s,1))
        return d
    m=min(k,n-1)
    ds=end(list(range(0,m+1)),-1); de=end(list(range(n-1,n-2-m,-1)),1)
    tu=T._turns(t); dt=[wdist(c,t.frames[i],*t.xy[i]) for i in tu]
    xs=[p[0] for p in t.xy];ys=[p[1] for p in t.xy]
    return dict(f0=t.start,f1=t.end,n=n,score=float(np.mean(t.scores)),reg=float(np.mean([c.region(f,*q) for f,q in zip(t.frames,t.xy)])),ds=ds,de=de,turns=[(t.frames[i],round(d,2)) for i,d in zip(tu,dt)],
      dmin=min(wdist(c,f,*q,0) for f,q in zip(t.frames,t.xy)),ext=math.hypot(max(xs)-min(xs),max(ys)-min(ys)),xy=list(zip(t.frames,t.xy)))
def run(k):
    c=rt.Ctx(k);c.byf={f.frame:[p for p in f.players if p.player_id] for f in c.pl.frames};fps=c.meta.fps
    cap={}
    class TT(ShuttleTracker):
        def _select(s,ts):
            cap['ts']=ts; r=super()._select(ts); cap['ch']=r; return r
    T=TT(fps,c.meta.width,c.meta.height,rt.S.tracker_config({},.2),region=c.region,support=c.support,rest=c.rest,strict_support=c.strict,held=c.held)
    T.track(c.cands,c.meta.frame_count); out=[]
    W=[((a-1)*fps,(b+1.5)*fps) for a,b in TR[k]['windows']]
    for t in cap['ch']:
        if any(a<=t.end and t.start<=b for a,b in W):
            out.append(dict(key=k,win=True,**feat(c,T,t)))
    return k,out,len(cap['ch'])
if __name__=='__main__':
    al=[]
    with Pool(6) as p:
        for k,o,n in p.imap(run,rt.IDS):
            al+=o; 
            nc=[x for x in o if x['ds']>.6 and x['de']>.6]
            print(k,'kept in win',len(o),'all',n,'no end contact',len(nc),'frames',sum(x['n'] for x in nc),'/',sum(x['n'] for x in o),'uncontacted turns',sum(1 for x in o if any(d>.8 for _,d in x['turns'])),flush=True)
    pickle.dump(al,open(rt.B+'/inputs/feats.pkl','wb'))
