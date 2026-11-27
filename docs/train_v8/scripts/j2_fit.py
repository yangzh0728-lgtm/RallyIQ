"""j2_fit.py OUTDIR: audio layer over v3b (see PLAN.md J2) -> OUTDIR/lovo_<key>.json (+ full.json), each a HitScorer spec
of kind 'audio_stack' wrapping the bundled v3b members."""
import os,sys,json,pickle,numpy as np
R=os.path.expanduser('~/Desktop/RallyIQ/train_v8/repo'); sys.path[:0]=[R,R+'/scripts']
from rallyiq.events.learned import ALL_FEATURES,HitScorer
from sklearn.linear_model import LogisticRegression
out=sys.argv[1]; os.makedirs(out,exist_ok=True)
am=pickle.load(open(os.path.expanduser('~/Desktop/RallyIQ/train_v8/ckpt/cands_PA/am_msc0.15.pkl'),'rb'))
QA={'own_20260529_0003','own_20260529_0004','own_20260529_0006','own_20260529_0008','own_20260906_0615','own_20260906_0620'}
LOVO=os.path.expanduser('~/Desktop/RallyIQ/train_v4/ckpt/lovo_B5')
AF=['a_peak','a_off','a_count']; ia=[ALL_FEATURES.index(f) for f in AF]; ih=ALL_FEATURES.index('has_audio')
def base(k): p=f'{LOVO}/lovo_{k}.json'; return json.load(open(p if os.path.exists(p) else f'{LOVO}/full.json'))
def rows(clips,scorer_of):
    X=[];y=[];w=[]
    for c in clips:
        m=(c.w>0)&(c.X[:,ih]>0)
        if not m.any(): continue
        p=np.clip(HitScorer(scorer_of(c.cid)).predict(c.X[m]),1e-4,1-1e-4)
        X.append(np.c_[np.log(p/(1-p)),c.X[m][:,ia]]); y.append(c.y[m]); w.append(c.w[m])
    return np.vstack(X),np.concatenate(y),np.concatenate(w)
def fit(clips,scorer_of):
    X,y,w=rows(clips,scorer_of); m=LogisticRegression(C=1.0,max_iter=2000).fit(X,y,sample_weight=w)
    return {'w':[float(m.intercept_[0])]+m.coef_[0].tolist()}
AM=[c for c in am if c.cid not in QA]
# training rows use each training video's own out-of-fold v3b (lovo_<video>) so the layer sees honest base scores
for v in [c.cid for c in AM]:
    L=fit([c for c in AM if c.cid!=v],lambda k:base(k)); spec={**base(v),'audio_stack':L}
    json.dump(spec,open(f'{out}/lovo_{v}.json','w')); print(v,np.round(L['w'],2))
L=fit(AM,lambda k:base(k)); json.dump({**base('full'),'audio_stack':L},open(f'{out}/full.json','w')); print('full',np.round(L['w'],2))
