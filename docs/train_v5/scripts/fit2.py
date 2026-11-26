import sys; sys.path.insert(0,'/Users/mac/Desktop/RallyIQ/train_v5/scripts')
import pickle,rt,numpy as np,json,os,math
from multiprocessing import Pool
from lab import F,O
from rallyiq.perception.shuttle import ownership as OW
from rallyiq.perception.shuttle.tracker import ShuttleTracker
MG=(0,.2,.4,.6)
class X(OW.CourtOwnership):
    def strip2(self,frame,span):
        k=(frame,span)
        if not hasattr(self,'c2'): self.c2={}
        if k in self.c2: return self.c2[k]
        w=span*self.fps;import bisect
        i,j=bisect.bisect_left(self.frames,frame-w),bisect.bisect_right(self.frames,frame+w);st=max(1,(j-i)//8);sd=set();n=0
        lo=[math.inf]*4;hi=[-math.inf]*4
        for f in self.frames[i:j:st]:
            n=max(n,len(self.rows[f]))
            for _p,s,x1,x2,h,_w in self.rows[f]:
                sd.add(s)
                for q,m in enumerate(MG): lo[q]=min(lo[q],x1-m*h);hi[q]=max(hi[q],x2+m*h)
        r=None if not {'near','far'}<=sd else (np.array(lo)-(OW.MISSING_PAD*self.width if n<self.expected else 0),np.array(hi)+(OW.MISSING_PAD*self.width if n<self.expected else 0))
        self.c2[k]=r; return r
    def fx(self,frames,xy,turns):
        out={}
        for span in (.4,.8,1.2):
            e=[]
            for f,(x,_) in zip(frames,xy):
                s=self.strip2(f,span)
                if s is None: continue
                e.append(np.maximum(np.maximum(s[0]-x,x-s[1]),0)/self.width)
            if not e or len(e)<.5*len(frames): return None
            e=np.array(e)
            for q,m in enumerate(MG):
                out[f'o{m}s{span}']=(e[:,q]>0).mean();out[f'm{m}s{span}']=e[:,q].mean();out[f'x{m}s{span}']=e[:,q].max()
        g=sum(self.contact(frames[i],*xy[i]) for i in turns)
        out.update(bad=min(len(turns)-g,4),good=min(g,4),cs=self.contact(frames[0],*xy[0]),ce=self.contact(frames[-1],*xy[-1]),dur=len(frames)/self.fps)
        return out
def ss(a):
    ad=a; vid=ad.split('/')[-2]
    c=rt.Ctx('x',ad,f'{rt.B}/inputs/ss_cands/shuttleset_train_v3/{vid}.json');o=X(c.pl,c.meta.fps,c.meta.width);cap={}
    sx=c.meta.width/c.size[0]
    class TT(ShuttleTracker):
        def _select(s,ts): r=super()._select(ts);cap['r']=r;return r
    T=TT(c.meta.fps,c.meta.width,c.meta.height,rt.S.tracker_config({},.2),region=c.region,support=c.support,rest=c.rest,strict_support=c.strict,held=c.held)
    T.track({f:[(x*sx,y*sx,s) for x,y,s in v] for f,v in c.cands.items()},c.meta.frame_count);out=[]
    for t in cap['r']:
        if len(t.frames)<4: continue
        f=o.fx(t.frames,t.xy,T._hard_turns(t))
        if f: out.append((vid,-1,0,f,len(t.frames)))
    return out
def own(k):
    al=pickle.load(open(rt.B+'/inputs/feats.pkl','rb'));c=rt.Ctx(k);T=ShuttleTracker(c.meta.fps,1920,1080);from rallyiq.perception.shuttle.tracker import _Tracklet;o=X(c.pl,c.meta.fps,c.meta.width);out=[]
    for i,t in enumerate(al):
        if t['key']!=k or t['n']<4: continue
        fr=[f for f,_ in t['xy']];q=[q for _,q in t['xy']];f=o.fx(fr,q,T._hard_turns(_Tracklet(fr,q,[1]*len(fr),fr)))
        if f: out.append((k,i,int(i in F),f,t['n']))
    return out
if __name__=='__main__':
    idx=json.load(open(rt.H+'/train_v3/inputs/public/shuttleset_train_v3/index.json'))
    with Pool(5) as p:
        R=sum(p.map(own,list(rt.IDS)),[]);print(len(R));R+=sum(p.map(ss,[e['artifacts'] for e in idx.values() if 'artifacts' in e]),[]);print(len(R))
    pickle.dump(R,open(rt.B+'/inputs/fit2.pkl','wb'))
