"""qmap.py SRC DST: map ft1 candidate scores onto the shipped detector's score distribution (monotone quantile map fitted on
all candidates >= 0.3 of the 17 own videos' labelled windows, label-free) -> DST/<key>.json; writes DST/qmap.json"""
import sys,json,os,numpy as np
s,d=sys.argv[1],sys.argv[2]; os.makedirs(d,exist_ok=True); B='/Users/mac/Desktop/RallyIQ/train_v5/inputs/cands'
A=[];F=[]
for n in os.listdir(s):
    if not n.endswith('.json') or not os.path.exists(f'{B}/{n}'): continue
    a=json.load(open(f'{s}/{n}'))['frames']; b=json.load(open(f'{B}/{n}'))['frames']
    for f,v in a.items():
        if v['selected']==-1 and f in b:  # re-detected frames only
            F+=[c[2] for c in v['candidates'] if c[2]>=.3]; A+=[c[2] for c in b[f]['candidates'] if c[2]>=.3]
q=np.linspace(0,1,201); fq=np.quantile(F,q); aq=np.quantile(A,q)
json.dump({'ft1':fq.tolist(),'base':aq.tolist(),'n_ft1':len(F),'n_base':len(A)},open(f'{d}/qmap.json','w'))
print(len(F),len(A),'ft1 q10/50/90',np.quantile(F,[.1,.5,.9]).round(3),'base',np.quantile(A,[.1,.5,.9]).round(3))
for n in os.listdir(s):
    if not n.endswith('.json'): continue
    doc=json.load(open(f'{s}/{n}'))
    for v in doc['frames'].values():
        v['candidates']=[[c[0],c[1],round(float(np.interp(c[2],fq,aq)) if c[2]>=.3 else c[2]*aq[0]/.3,4)] for c in v['candidates']]
    json.dump(doc,open(f'{d}/{n}','w'))
