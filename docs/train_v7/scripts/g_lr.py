"""g_lr.py [FEATS]: dead-ball filter = logistic regression on rally-context features, trained on the 11 am7 videos only.
Nested leave-one-video-out on am7 (threshold chosen on the training videos' own out-of-fold scores); qa6 and fresh scored
with the model fitted on all 11 and the threshold chosen the same way. Drops a hit when p(dead) > threshold."""
import json,sys,random,numpy as np
from sklearn.linear_model import LogisticRegression
R=json.load(open('ckpt/g_rows.json')); BASE=json.load(open('ckpt/g_base.json'))
FE=sys.argv[1].split(',') if len(sys.argv)>1 else list(R[0]['f'])
def XY(rows): return np.array([[r['f'][f] for f in FE] for r in rows]),np.array([r['lab'] in('neg','unl') for r in rows],float)
import os
X_=json.load(open('ckpt/g_rows_x.json')) if os.environ.get('AUG') else []
AM=[r for r in R if r['set']=='am7' and r['lab']!='ign']+X_; vids=sorted({r['k'] for r in AM if r['set']=='am7'})
CS=[float(c) for c in os.environ.get('CS',os.environ.get('C','0.3')).split(',')]
def fit(rows,C=None):
    C=C if C is not None else CS[0]
    X,y=XY(rows); mu,sd=X.mean(0),X.std(0)+1e-6; m=LogisticRegression(C=C,class_weight=None,max_iter=2000).fit((X-mu)/sd,y); return lambda rr: m.predict_proba((XY(rr)[0]-mu)/sd)[:,1]
def oof(rows,C=None):
    p=np.zeros(len(rows)); vs=sorted({r['k'] for r in rows})
    for v in vs:
        tr=[r for r in rows if r['k']!=v]; te=[i for i,r in enumerate(rows) if r['k']==v]
        p[te]=fit(tr,C)([rows[i] for i in te])
    return p
def counts(rows,p,thr,keys):
    c={k:dict(BASE[k]) for k in keys}
    for r,q in zip(rows,p):
        if q>thr and r['k'] in c and r['set']!='am7x':
            if r['lab']=='pos': c[r['k']]['tp']-=1; c[r['k']]['fn']+=1
            elif r['lab'] in('neg','unl'): c[r['k']]['fp']-=1
    return c
def F(c): t=[sum(v[x] for v in c.values()) for x in('tp','fp','fn')]; return 2*t[0]/max(2*t[0]+t[1]+t[2],1)
TH=[.5,.6,.7,.8,.9,1.01]
def pick(rows,p,keys): return max(TH,key=lambda th:(F(counts(rows,p,th,keys)),th))
def pickCT(rows,keys):
    best=None
    for C in CS:
        p=oof(rows,C); th=pick(rows,p,keys); f=F(counts(rows,p,th,keys))
        if best is None or f>best[0]+1e-9: best=(f,C,th)
    return best[1],best[2]
# nested LOVO on am7
res={}
for v in vids:
    tr=[r for r in AM if r['k']!=v]; Cv,th=pickCT(tr,sorted({r['k'] for r in tr if r['set']=='am7'})); te=[r for r in R if r['k']==v]
    res[v]=counts(te,fit(tr,Cv)(te),th,[v])[v]; res[v]['th']=(Cv,th)
CA,thA=pickCT(AM,vids); full=fit(AM,CA); print('full C',CA)
for s in ('qa6','fresh'):
    rows=[r for r in R if r['set']==s]; ks=sorted({r['k'] for r in rows}|({k for k in BASE if k.startswith('own_20260529_000') and k[-1] in '579'} if s=='fresh' else set()))
    for k,v in counts(rows,full(rows),thA,ks).items(): res[k]=v
json.dump(res,open('ckpt/g_res.json','w'))
QA=[k for k in BASE if k in {'own_20260529_0003','own_20260529_0004','own_20260529_0006','own_20260529_0008','own_20260906_0615','own_20260906_0620'}]
FR=['own_20260529_0005','own_20260529_0007','own_20260529_0009']
for name,ks in (('am7 (nested LOVO)',vids),('qa6',QA),('all 17',vids+QA),('fresh',FR)):
    print(f"{name:18s} base {F({k:BASE[k] for k in ks}):.4f} -> {F({k:res[k] for k in ks}):.4f}")
print('thresholds',[res[v]['th'] for v in vids],'full',thA)
rng=random.Random(0); ks=vids+QA
d=sorted(F({k:res[k] for k in s})-F({k:BASE[k] for k in s}) for s in ([rng.choice(ks) for _ in ks] for _ in range(5000)))
print('all 17 diff CI [%.4f, %.4f]'%(d[125],d[4875]))
m=fit(AM); 
X,y=XY(AM); mu,sd=X.mean(0),X.std(0)+1e-6; mm=LogisticRegression(C=float(os.environ.get('C',0.3)),max_iter=2000).fit((X-mu)/sd,y)
print('coef',dict(zip(FE,(mm.coef_[0]).round(2))))
for s in ('am7','qa6','fresh'):
    rows=[r for r in R if r['set']==s and r['lab']!='ign']; p=full(rows)
    print(s,'dropped',[(r['k'][-4:],r['t'],r['lab']) for r,q in zip(rows,p) if q>thA])
