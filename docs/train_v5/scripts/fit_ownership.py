import sys; sys.path.insert(0,'/Users/mac/Desktop/RallyIQ/train_v5/scripts')
import pickle,rt,numpy as np
from lab import F,O
from rallyiq.perception.shuttle import ownership as OW
from rallyiq.perception.shuttle.tracker import ShuttleTracker,_Tracklet,TrackerConfig
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut,cross_val_predict
from sklearn.metrics import roc_auc_score
al=pickle.load(open(rt.B+'/inputs/feats.pkl','rb'));C={};X=[];Y=[];G=[];I=[];W=[]
for i,t in enumerate(al):
    k=t['key']
    if k not in C: c=rt.Ctx(k);C[k]=(c,OW.CourtOwnership(c.pl,c.meta.fps,c.meta.width))
    fr=[f for f,_ in t['xy']];ix={f:j for j,f in enumerate(fr)}
    ft=C[k][1].features(fr,[q for _,q in t['xy']],[ix[f] for f,_ in t['turns']])
    if ft is None or t['n']<4: continue
    X.append([ft[n] for n in OW.FEATURES]);Y.append(int(i in F));G.append(k);I.append(i);W.append(1 if i in F or i in O else .5)
X=np.array(X);Y=np.array(Y);G=np.array(G);I=np.array(I);W=np.array(W)
sd=X.std(0)+1e-6
from sklearn.pipeline import make_pipeline
def fit(tr): 
    return LogisticRegression(C=.5).fit(X[tr]/sd,Y[tr],W[tr])
p=np.zeros(len(Y))
for g in set(G): p[G==g]=fit(G!=g).predict_proba(X[G==g]/sd)[:,1]
print('n',len(Y),Y.sum(),'LOVO auc',roc_auc_score(Y,p))
for th in(.2,.3,.4,.5,.6,.7,.8): print(th,'F caught',round((p[Y==1]>th).mean(),3),'frames',round(sum(al[i]['n'] for i in I[(Y==1)&(p>th)])/sum(al[i]['n'] for i in I[Y==1]),3),'ours dropped',round((p[Y==0]>th).mean(),4),'hand-O dropped',np.mean([p[j]>th for j in range(len(I)) if I[j] in O]).round(3))
m=fit(np.ones(len(Y),bool)); w=m.coef_[0]/sd; print('WEIGHTS',dict(zip(OW.FEATURES,w.round(3))),'BIAS',m.intercept_[0].round(3))
pickle.dump((I,p,Y),open(rt.B+'/inputs/own_p.pkl','wb'))
print([(I[j],G[j][-7:],al[I[j]]['n'],p[j].round(2)) for j in np.argsort(-p) if Y[j]==0][:40])
