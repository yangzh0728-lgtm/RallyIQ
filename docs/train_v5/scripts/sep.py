import sys; sys.path.insert(0,'/Users/mac/Desktop/RallyIQ/train_v5/scripts')
import pickle,rt,numpy as np,math
from lab import F,O
al=pickle.load(open(rt.B+'/inputs/feats.pkl','rb')); C={}
def span(c,t):
    fps=c.meta.fps;w=round(.5*fps);ex=[]
    for f,(x,y) in t['xy']:
        lo=1e9;hi=-1e9;
        for g in range(f-w,f+w+1,max(1,w//8)):
            for p in c.byf.get(g,()) or c.byf.get(g+1,()):
                h=p.bbox[3]-p.bbox[1]
                a,b=p.bbox[0]-.4*h,p.bbox[2]+.4*h
                lo=min(lo,a);hi=max(hi,b)
        if hi<lo: continue
        ex.append(max(lo-x,x-hi,0)/c.meta.width)
    return (np.mean([e>0 for e in ex]),max(ex)) if ex else (0,0)
R=[]
for i,t in enumerate(al):
    k=t['key']
    if k not in C: C[k]=c=rt.Ctx(k);c.byf={f.frame:[p for p in f.players if p.player_id] for f in c.pl.frames}
    t['out'],t['outmax']=span(C[k],t)
pickle.dump(al,open(rt.B+'/inputs/feats2.pkl','wb'))
def lab(i): return 'F' if i in F else ('O' if i in O else ('o'))
import collections
for L in 'FOo':
    X=[t for i,t in enumerate(al) if lab(i)==L]
    print(L,len(X),'out>.15',np.mean([t['out']>.15 for t in X]),'outmax q',np.quantile([t['outmax'] for t in X],[.1,.25,.5,.75,.9,.97]).round(3),'out q',np.quantile([t['out'] for t in X],[.1,.25,.5,.75,.9,.97]).round(2),'reg',np.quantile([t['reg'] for t in X],[.1,.5,.9]).round(2))
print('F low out',[(i,al[i]['key'][-4:],al[i]['n'],round(al[i]['out'],2),round(al[i]['outmax'],3)) for i in sorted(F) if al[i]['out']<.2])
print('O high out',[(i,al[i]['key'][-4:],al[i]['n'],round(al[i]['out'],2),round(al[i]['outmax'],3)) for i in sorted(O) if al[i]['out']>.1])
print('o high out',[(i,al[i]['key'][-7:],al[i]['n'],round(al[i]['out'],2),round(al[i]['outmax'],3)) for i in range(len(al)) if lab(i)=='o' and al[i]['out']>.3])
