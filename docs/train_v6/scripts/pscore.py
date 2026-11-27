"""pscore.py INPUTS KEY... : bundled v3b (LOVO lovo_B5) hit F1 per video on a subset."""
import sys,json
from pathlib import Path
from dataclasses import replace
R=Path('~/Desktop/RallyIQ/train_v6/repo').expanduser(); sys.path[:0]=[str(R),str(R/'scripts')]
import amateur_hits as AH
from rallyiq.events.config import EventsConfig
P={}
lovo=Path('~/Desktop/RallyIQ/train_v4/ckpt/lovo_B5').expanduser(); T=[0,0,0]
for c in AH.load_clips(Path(sys.argv[1]).expanduser(),Path(__import__('os').environ.get('LAB','~/Desktop/RallyIQ/train_v5/labels/amateur_v4.json')).expanduser(),only=sys.argv[2:]):
    lp=lovo/f'lovo_{c.cid}.json'; p=c.predict_full(replace(EventsConfig(),**{**AH.AMATEUR_BASE,'hit_scorer':str(lp if lp.exists() else lovo/'full.json')}))
    P[c.cid]=p; s=AH.score_clip(c,p); T=[T[0]+s['tp'],T[1]+s['fp'],T[2]+s['fn']]; print(c.cid,s['tp'],s['fp'],s['fn'],round(2*s['tp']/max(2*s['tp']+s['fp']+s['fn'],1),3),flush=True)
json.dump(P,open(Path('~/Desktop/RallyIQ/train_v6/ckpt/preds').expanduser()/('p_'+Path(sys.argv[1]).name+'.json'),'w'))
print('ALL',T,round(2*T[0]/(2*T[0]+T[1]+T[2]),3))
