import sys; sys.path.insert(0,'/Users/mac/Desktop/RallyIQ/train_v5/scripts')
import pickle,rt,numpy as np
from lab import F,O
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
R=pickle.load(open(rt.B+'/inputs/fit2.pkl','rb'))
Y=np.array([r[2] for r in R]);G=np.array([r[0] if r[1]>=0 else 'ss'+str(hash(r[0])%5) for r in R]);SS=np.array([r[1]<0 for r in R]);N=np.array([r[4] for r in R])
hand=np.array([r[1] in O for r in R])
def ev(S,ws=.4,C=.5,verbose=False):
    X=np.array([[float(r[3][s]) for s in S] for r in R]);sd=X.std(0)+1e-6;W=np.where(SS,ws,np.where((Y==1)|hand,1,.5))
    p=np.zeros(len(Y))
    for g in set(G):
        m=LogisticRegression(C=C,max_iter=500).fit(X[G!=g]/sd,Y[G!=g],W[G!=g]);p[G==g]=m.predict_proba(X[G==g]/sd)[:,1]
    o=~SS;res=[]
    for th in(.3,.5,.7): res.append((th,f"F {((p[(Y==1)]>th)*N[Y==1]).sum()/N[Y==1].sum():.3f} own {((p[o&(Y==0)]>th)*N[o&(Y==0)]).sum()/N[o&(Y==0)].sum():.4f} hand {((p[hand]>th)*N[hand]).sum()/N[hand].sum():.3f} ss {((p[SS]>th)*N[SS]).sum()/N[SS].sum():.4f}"))
    print(S,'auc',round(roc_auc_score(Y[o],p[o]),4),round(roc_auc_score(Y,p),4));[print('   ',*r) for r in res]
    m=LogisticRegression(C=C,max_iter=500).fit(X/sd,Y,W);return dict(zip(S,(m.coef_[0]/sd).round(3))),m.intercept_[0].round(3),p
T=['bad','good','cs','ce']
for sp in(.4,.8,1.2):
    ev([f'o0s{sp}',f'o0.4s{sp}',f'm0s{sp}',f'x0.4s{sp}']+T); ev([f'o0.2s{sp}',f'm0.2s{sp}',f'x0.4s{sp}']+T); ev([f'o0.2s{sp}',f'm0.2s{sp}',f'm0.6s{sp}',f'x0.6s{sp}']+T)
ev(['m0s0.4','m0.2s0.8','x0.4s0.4','o0.2s0.4','o0.2s0.8']+T); ev(['m0s0.4','m0.2s0.8','x0.4s0.4','o0.2s0.4','o0.2s0.8','dur']+T)
print('=== with concurrency')
import json
it=json.load(open(rt.H+'/train_v3/inputs/public/shuttleset_train_v3/index.json'));v2={e['artifacts'].split('/')[-2]:k for k,e in it.items() if 'artifacts' in e};cc=json.load(open(rt.B+'/inputs/conc.json'))
for r in R: r[3]['lc']=np.log(cc[r[0] if r[1]>=0 else v2[r[0]]]+.02)
for ws in(.2,.5):
  for S in (['o0s0.4','o0.4s0.4','m0s0.4','x0.4s0.4'],['m0s0.4','m0.2s0.8','x0.4s0.4','o0.2s0.4','o0.2s0.8'],['o0.2s0.4','m0.2s0.4','x0.4s0.4']):
    r=ev(S+T+['lc'],ws); print(r[:2])
