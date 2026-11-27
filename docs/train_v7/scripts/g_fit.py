"""g_fit.py OUT: fit the dead-ball model on all 11 am7 videos (ckpt/g_rows.json), threshold from out-of-fold scores -> OUT json."""
import json,sys,numpy as np
sys.argv+=[] ; exec(open('scripts/g_lr.py').read().split('# nested LOVO on am7')[0].replace("FE=sys.argv[1].split(',') if len(sys.argv)>1 else list(R[0]['f'])","FE=list(R[0]['f'])"))
X,y=XY(AM); mu,sd=X.mean(0),X.std(0)+1e-6; m=LogisticRegression(C=CS[0],max_iter=2000).fit((X-mu)/sd,y)
th=pick(AM,oof(AM),vids)
json.dump({'kind':'logistic','features':FE,'mean':mu.round(6).tolist(),'std':sd.round(6).tolist(),'coef':m.coef_[0].round(6).tolist(),'intercept':round(float(m.intercept_[0]),6),
           'threshold':th,'swing_model':'auto','trained_on':vids,'n':{'pos':int((y==0).sum()),'dead':int(y.sum())},
           'note':'train_v7 phase G: logistic regression, C chosen by nested LOVO (1.0), labels amateur_v6 (am7) + 91 dead-time examples; threshold chosen on out-of-fold scores'},open(sys.argv[1],'w'),indent=1)
print('threshold',th,'n',len(y),int(y.sum()))
