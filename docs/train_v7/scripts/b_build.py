"""b_build.py -> ckpt/b/spec.json: own flight snippets (train/val), hard positives, TrackNet rehearsal clips.
Own labels: v5 track points inside hit-to-hit flights of the labelled windows (qa6 never used), minus frames where a
hand-labelled next-court tracklet is also in the air, plus my accepted hard positives (labels/shuttle_v6/hp_decisions.txt)."""
import json,os,pickle,random,glob,csv,sys
sys.path.insert(0,'/Users/mac/Desktop/RallyIQ/train_v5/scripts')
from lab import F
B5='/Users/mac/Desktop/RallyIQ/train_v5'; B6='/Users/mac/Desktop/RallyIQ/train_v6'
L=json.load(open(f'{B5}/labels/amateur_v4.json'))
QA6={'own_20260529_0003','own_20260529_0004','own_20260529_0006','own_20260529_0008','own_20260906_0615','own_20260906_0620'}
VALV={'own_20260320_0239','own_20260906_0622'}
acc=set()
for l in open(f'{B6}/labels/shuttle_v6/hp_decisions.txt'):
    if l.startswith('mos') and ' accept ' in l: acc.update(map(int,l.split()[2:]))
HP={}
for c in json.load(open(f'{B6}/labels/shuttle_v6/hp_cands.json')):
    if c['id'] in acc: HP.setdefault(c['k'],{})[c['frame']]=(round(c['x'],1),round(c['y'],1))
al=pickle.load(open(f'{B5}/inputs/feats.pkl','rb')); foreign={}
for i in F:
    t=al[i]; foreign.setdefault(t['key'],set()).update(g for g,_ in t['xy'])
src=lambda k: next(s for s in (f'/Users/mac/Desktop/RallyIQ/train_v4/inputs/sources/{k}.mp4',f'{B5}/inputs/sources/{k}.mp4',f'/Users/mac/Desktop/RallyIQ/train_v4/inputs/new_runs/{k}/source.mp4') if os.path.exists(s))
own={'train':[],'val':[]}
for k,hs in L['hits'].items():
    if k in QA6: continue
    S=json.load(open(f'{B5}/inputs_C/new_runs/{k}/shuttle.json')); fps=json.load(open(f'{B5}/inputs_C/new_runs/{k}/probe.json'))['fps']
    det={p['frame']:(round(p['x'],1),round(p['y'],1)) for p in S['points'] if p.get('source','detected')=='detected'}
    pos=sorted(h['t'] for h in hs if h['status'] in('reviewed_sure','reviewed_unsure'))
    cut=pos[0]+0.75*(pos[-1]-pos[0])
    for a,b in zip(pos,pos[1:]):
        if b-a>2.5: continue
        lab={}; hp=[]
        for f in range(round(a*fps)+2,round(b*fps)-1):
            if f in foreign.get(k,()): continue
            if f in det: lab[f]=det[f]
            elif f in HP.get(k,{}): lab[f]=HP[k][f]; hp.append(f)
        # keep a track point only where it lies on a smooth path (quadratic through up to 4 detected neighbours
        # on each side within 5 px): my check of 48 random points found 5 off the shuttle (a shirt logo, a second
        # shuttle) and all 5 had residuals above 5 px
        import numpy as np
        fs=sorted(f for f in lab if f not in hp); keep={}
        for i,f in enumerate(fs):
            nb=[g for g in fs[max(0,i-4):i+5] if g!=f]
            if len(nb)<4: continue
            T=np.array(nb,float); X=np.array([lab[g][0] for g in nb]); Y=np.array([lab[g][1] for g in nb])
            if np.hypot(np.polyval(np.polyfit(T,X,2),f)-lab[f][0],np.polyval(np.polyfit(T,Y,2),f)-lab[f][1])<=5: keep[f]=lab[f]
        lab={**keep,**{f:lab[f] for f in hp}}
        if len(lab)<5: continue
        sp='val' if (k in VALV or a>=cut) else 'train'
        own[sp].append({'id':f'{k}@{a:.2f}','video':src(k),'fps':fps,'key':k,'labels':{str(f):v for f,v in lab.items()},'hp':hp})
tn=[]
root='/Users/mac/rallyiq-data/public/tracknet_ds/Professional'
for m in sorted(os.listdir(root)):
    if m.startswith('.'): continue
    for cf in sorted(glob.glob(f'{root}/{m}/csv/*_ball.csv')):
        if os.path.basename(cf).startswith('._'): continue
        v=cf.replace('/csv/','/video/').replace('_ball.csv','.mp4')
        if os.path.exists(v): tn.append((m,cf,v))
rng=random.Random(0); rng.shuffle(tn); reh={'train':[],'val':[]}
for j,(m,cf,v) in enumerate(tn[:70]):
    rows=list(csv.DictReader(open(cf))); s0=rng.randrange(0,max(1,len(rows)-150))
    lab={str(int(r['Frame'])):(None if r['Visibility']=='0' else (float(r['X']),float(r['Y']))) for r in rows[s0:s0+150]}
    reh['val' if j>=60 else 'train'].append({'id':f'tn_{m}_{os.path.basename(cf)[:-9]}','video':v,'fps':30.0,'labels':lab,'hp':[]})
json.dump({'own':own,'tn':reh},open(f'{B6}/ckpt/b/spec.json','w'))
for s in own: print('own',s,len(own[s]),'flights',sum(len(x['labels']) for x in own[s]),'frames',sum(len(x['hp']) for x in own[s]),'hp')
for s in reh: print('tn',s,len(reh[s]),sum(len(x['labels']) for x in reh[s]))
