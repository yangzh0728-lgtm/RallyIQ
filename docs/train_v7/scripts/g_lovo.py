"""g_lovo.py: per am7 video, the dead-ball model fitted on the other 10 (threshold from their out-of-fold scores) ->
ckpt/dead_lovo/lovo_<key>.json; full.json = all 11 (used for qa6 / fresh)."""
import json,numpy as np
exec(open('scripts/g_lr.py').read().split('# nested LOVO on am7')[0].replace("FE=sys.argv[1].split(',') if len(sys.argv)>1 else list(R[0]['f'])","FE=list(R[0]['f'])"))
def export(rows,ks,path):
    X,y=XY(rows); mu,sd=X.mean(0),X.std(0)+1e-6; m=LogisticRegression(C=0.3,max_iter=2000).fit((X-mu)/sd,y)
    json.dump({'kind':'logistic','features':FE,'mean':mu.tolist(),'std':sd.tolist(),'coef':m.coef_[0].tolist(),'intercept':float(m.intercept_[0]),'threshold':pick(rows,oof(rows),ks),'swing_model':'auto'},open(path,'w'))
for v in vids:
    tr=[r for r in AM if r['k']!=v]; export(tr,sorted({r['k'] for r in tr}),f'ckpt/dead_lovo/lovo_{v}.json')
export(AM,vids,'ckpt/dead_lovo/full.json'); print('ok')
