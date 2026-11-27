"""mine_hp.py: hard-positive candidates = short gaps (1-8 frames) of the v5 track inside hit-to-hit flights of the
labelled windows; position from a quadratic fit to up to 4 detected points on each side.
-> labels/shuttle_v6/hp_cands.json [{k,frame,x,y,gap,id}] and review/hp/mos_NN.jpg (48 crops, mark = estimate)."""
import json,os,cv2,numpy as np,random
B5='/Users/mac/Desktop/RallyIQ/train_v5'; B6='/Users/mac/Desktop/RallyIQ/train_v6'
L=json.load(open(f'{B5}/labels/amateur_v4.json'))
QA6={'own_20260529_0003','own_20260529_0004','own_20260529_0006','own_20260529_0008','own_20260906_0615','own_20260906_0620'}
out=[]
for k,hs in L['hits'].items():
    if k in QA6: continue
    S=json.load(open(f'{B5}/inputs_C/new_runs/{k}/shuttle.json')); fps=json.load(open(f'{B5}/inputs_C/new_runs/{k}/probe.json'))['fps']
    det={p['frame']:(p['x'],p['y']) for p in S['points'] if p.get('source','detected')=='detected'}
    pos=sorted(h['t'] for h in hs if h['status'] in('reviewed_sure','reviewed_unsure'))
    for a,b in zip(pos,pos[1:]):
        if b-a>2.5: continue
        fr=list(range(round(a*fps)+2,round(b*fps)-1)); have=[f for f in fr if f in det]
        for p,q in zip(have,have[1:]):
            g=q-p-1
            if not 1<=g<=8: continue
            L_=[f for f in have if f<=p][-4:]; R_=[f for f in have if f>=q][:4]
            T=np.array(L_+R_,float); X=np.array([det[f][0] for f in L_+R_]); Y=np.array([det[f][1] for f in L_+R_])
            deg=2 if len(T)>=5 else 1
            cx,cy=np.polyfit(T,X,deg),np.polyfit(T,Y,deg)
            for f in range(p+1,q):
                out.append({'k':k,'frame':f,'x':float(np.polyval(cx,f)),'y':float(np.polyval(cy,f)),'gap':g})
random.Random(0).shuffle(out)
for i,o in enumerate(out): o['id']=i
json.dump(out,open(f'{B6}/labels/shuttle_v6/hp_cands.json','w'))
print(len(out),{k:sum(o['k']==k for o in out) for k in {o['k'] for o in out}})
