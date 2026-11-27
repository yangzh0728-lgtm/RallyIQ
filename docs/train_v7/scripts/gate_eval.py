"""gate_eval.py: hit-anchored display gate on the v5 tracks: labelled foreign / ours tracklet frames kept."""
import os,sys,json,pickle
from pathlib import Path
from dataclasses import replace
R=Path('~/Desktop/RallyIQ/train_v6/repo').expanduser(); sys.path[:0]=[str(R),str(R/'scripts'),'/Users/mac/Desktop/RallyIQ/train_v5/scripts']
import amateur_hits as AH
from lab import F,O
from rallyiq.events.config import EventsConfig
B5=Path('~/Desktop/RallyIQ/train_v5').expanduser(); B6=Path('~/Desktop/RallyIQ/train_v6').expanduser()
al=pickle.load(open(B5/'inputs/feats.pkl','rb'))
lovo=Path('~/Desktop/RallyIQ/train_v4/ckpt/lovo_B5').expanduser()
cache=B6/'ckpt/rallies_v6A2.json'
if cache.exists(): RL=json.load(open(cache))
else:
    RL={}
    for c in AH.load_clips(B5/'inputs_C',B5/'labels/amateur_v4.json'):
        lp=lovo/f'lovo_{c.cid}.json'; cfg=replace(EventsConfig(),**{**AH.AMATEUR_BASE,'hit_scorer':str(lp if lp.exists() else lovo/'full.json')})
        RL[c.cid]=[[r.start_ms,r.end_ms,[h.t_ms for h in r.hits]] for r in c.run(cfg)]
    json.dump(RL,open(cache,'w'))
fps={k:json.load(open(B5/f'inputs_C/new_runs/{k}/probe.json'))['fps'] for k in RL}
pts={k:{p['frame']:(p['x'],p['y']) for p in json.load(open(B5/f'inputs_C/new_runs/{k}/shuttle.json'))['points']} for k in RL}
LONG=False
def allowed(k,G,pre,post):
    iv=[]
    for s,e,hs in RL[k]:
        if not hs: continue
        hs=sorted(hs); iv.append((hs[0]-pre*1000,hs[0]))
        for a,b in zip(hs,hs[1:]):
            if b-a<=G*1000: iv.append((a,b))
            else: iv+= [(a,a+post*1000),(b-pre*1000,b)]
        iv.append((hs[-1],hs[-1]+post*1000))
        if LONG: iv=[(s-1e9 if j==0 else a,b) for j,(a,b) in enumerate(iv)]
    return iv
def score(G,pre,post):
    res={'F':[0,0],'O':[0,0],'all':[0,0]}
    for k in RL:
        iv=allowed(k,G,pre,post) if G else None; f=fps[k]
        ok=lambda fr: iv is None or any(a<=fr*1000/f<=b for a,b in iv)
        for i,t in enumerate(al):
            if t['key']!=k: continue
            S='F' if i in F else ('O' if i in O else None)
            if S is None: continue
            for g,q in t['xy']:
                r=pts[k].get(g)
                if r is not None and abs(r[0]-q[0])+abs(r[1]-q[1])<6: res[S][1]+=1; res[S][0]+=ok(g)
        n=len(pts[k]); res['all'][1]+=n; res['all'][0]+=sum(ok(g) for g in pts[k]) if iv is not None else n
    return res
print('no gate',score(None,0,0))
for G in (3,4,5):
    for pre,post in ((1.,2.5),(1.5,3.),(2.,3.5)): print(G,pre,post,score(G,pre,post))
