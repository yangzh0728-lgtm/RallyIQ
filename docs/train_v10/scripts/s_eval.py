"""s_eval.py TAG: stroke classifier vs court_v2 + residual on the same own rows (LOVO, per-video bootstrap) and fresh rows."""
import os,sys,json,random
import numpy as np
B=os.path.expanduser('~/Desktop/RallyIQ/train_v10'); tag=sys.argv[1]
d=np.load(f'{B}/ckpt/s_data.npz'); y,g,key,t,sure,n=d['y'],d['g'],d['key'],d['t'],d['sure'],d['n']
sys.path.insert(0,os.path.expanduser('~/Desktop/RallyIQ/train_v10/repo'))
from rallyiq.strokes import court as CM
CL=CM.get_model().classes
P={k:np.array(v) for k,v in json.load(open(f'{B}/ckpt/s_{tag}/preds.json')).items()}
base=json.load(open(f'{B}/ckpt/s_base_rows.json')); bmap={(r['key'],round(r['t'],3)):r for r in base}
rows=[]
for i in np.where(g=='own')[0]:
    r=bmap[(key[i],round(float(t[i]),3))]; rows.append((key[i],bool(sure[i]),CL[y[i]],CL[int(P[str(i)].argmax())],r['resid'],r['base']))
tr=[r for r in rows if r[1]]
acc=lambda rs,j: np.mean([r[2]==r[j] for r in rs])
print(f'own trusted (n={len(tr)}): court {acc(tr,5):.3f}  court+residual {acc(tr,4):.3f}  seq-classifier {acc(tr,3):.3f}')
print(f'own all typed (n={len(rows)}): court+residual {acc(rows,4):.3f}  seq {acc(rows,3):.3f}')
vids=sorted({r[0] for r in tr}); rng=random.Random(0); ds=[]
for _ in range(3000):
    s=[r for v in [rng.choice(vids) for _ in vids] for r in tr if r[0]==v]; ds.append(acc(s,3)-acc(s,4))
ds.sort(); print('seq - residual, trusted, 95%% CI over videos [%+.3f, %+.3f]'%(ds[75],ds[2925]))
kf=json.load(open(os.path.expanduser('~/Desktop/RallyIQ/train_v9/ckpt/k_fresh.json')))
res={r[1]:r[4] for r in kf['resid']}
fr=[(int(n[i]),CL[y[i]],CL[int(P[str(i)].argmax())],res.get(int(n[i])),bool(sure[i])) for i in np.where(g=='fresh')[0]]
m=[r for r in fr if r[3] is not None]
print(f'fresh (n={len(m)} matched): residual {np.mean([r[1]==r[3] for r in m]):.3f}  seq {np.mean([r[1]==r[2] for r in m]):.3f}; all {len(fr)} typed seq {np.mean([r[1]==r[2] for r in fr]):.3f}; sure seq {np.mean([r[1]==r[2] for r in fr if r[4]]):.3f}')
te=np.where(g=='ss_test')[0]
print(f"ShuttleSet test (n={len(te)}): seq own-fine-tuned {np.mean([y[i]==P[str(i)].argmax() for i in te]):.3f}  ShuttleSet-only {np.mean([y[i]==P[str(('ssonly',int(i)))].argmax() for i in te]):.3f}")
