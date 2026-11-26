import sys; sys.path.insert(0,'/Users/mac/Desktop/RallyIQ/train_v5/scripts')
import pickle,mosaic,rt,cv2,sheets,numpy as np
from lab import F,O
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
al=pickle.load(open(rt.B+'/inputs/feats2.pkl','rb'))
X,Y,G,I=pickle.load(open(rt.B+'/inputs/clf.pkl','rb')); S=['o0','o4','m0','m4','tb','tg','cs','ce']
M=np.array([[x[k] for k in S] for x in X]);lg=make_pipeline(StandardScaler(),LogisticRegression(C=.3,class_weight='balanced')).fit(M,Y);p=lg.predict_proba(M)[:,1]
its=[(I[q][0],round(p[q],2)) for q in np.argsort(-p) if p[q]>.4 and I[q][0] not in F and I[q][0] not in O]
its.sort(); print(len(its),its)
V={};ts=[]
for i,pp in its:
    t=al[i];k=t['key']
    if k not in V: V[k]=(sheets.Video(k),rt.Ctx(k))
    im=mosaic.tile(*V[k][:1],t,V[k][1],f'{i} p{pp} {k[-7:]}'); ts.append(im)
for k in range(0,len(ts),24):
    b=ts[k:k+24]
    while len(b)%4: b.append(np.zeros((270,480,3),np.uint8))
    cv2.imwrite(f'{rt.B}/review/tk/unl_{k//24}.jpg',cv2.vconcat([cv2.hconcat(b[j:j+4]) for j in range(0,len(b),4)]),[1,82])
