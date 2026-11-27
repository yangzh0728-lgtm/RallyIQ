"""fit_own6.py: v5 ownership features + 'active' (any of our players swinging during the tracklet) -> LOVO AUC."""
import os,sys; os.environ['RTREPO']=os.path.expanduser('~/Desktop/RallyIQ/train_v6/repo'); sys.path.insert(0,'/Users/mac/Desktop/RallyIQ/train_v5/scripts')
import pickle,rt,numpy as np
from lab import F,O
from rallyiq.perception.shuttle import ownership as OW
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
al=pickle.load(open(rt.B+'/inputs/feats.pkl','rb'));C={};X=[];Y=[];G=[];I=[];W=[];A=[]
for i,t in enumerate(al):
    k=t['key']
    if k not in C: c=rt.Ctx(k);C[k]=OW.CourtOwnership(c.pl,c.meta.fps,c.meta.width)
    fr=[f for f,_ in t['xy']];ix={f:j for j,f in enumerate(fr)}
    ft=C[k].features(fr,[q for _,q in t['xy']],[ix[f] for f,_ in t['turns']])
    if ft is None or t['n']<4: continue
    a=C[k].activity(fr[0],fr[-1])
    X.append([ft[n] for n in OW.FEATURES]);A.append(a);Y.append(int(i in F));G.append(k);I.append(i);W.append(1 if i in F or i in O else .5)
X=np.array(X);A=np.array(A);Y=np.array(Y);G=np.array(G);I=np.array(I);W=np.array(W)
pickle.dump((X,A,Y,G,I,W),open(os.path.expanduser('~/Desktop/RallyIQ/train_v6/ckpt/own6.pkl'),'wb'))
for th in (.5,1,1.5,2):
    print('active>=',th,'foreign',round((A[Y==1]>=th).mean(),3),'ours',round((A[Y==0]>=th).mean(),3))
for name,XX in (('v5',X),('v5+active',np.c_[X,np.minimum(A,3)]),('v5+idle',np.c_[X,(A<1).astype(float)])):
    sd=XX.std(0)+1e-6; p=np.zeros(len(Y))
    for g in set(G): p[G==g]=LogisticRegression(C=.5).fit(XX[G!=g]/sd,Y[G!=g],W[G!=g]).predict_proba(XX[G==g]/sd)[:,1]
    print(name,'auc',round(roc_auc_score(Y,p),4),[(th,round((p[Y==1]>th).mean(),3),round((p[Y==0]>th).mean(),4)) for th in (.5,.7)])
    m=LogisticRegression(C=.5).fit(XX/sd,Y,W); print('  w',(m.coef_[0]/sd).round(3),round(m.intercept_[0],3))
