"""rallygate.py: are the v5 'between rallies' rows outside the detected rally segments?"""
import sys,json
from pathlib import Path
from dataclasses import replace
R=Path('~/Desktop/RallyIQ/train_v6/repo').expanduser(); sys.path[:0]=[str(R),str(R/'scripts'),'/Users/mac/Desktop/RallyIQ/train_v5/scripts']
import amateur_hits as AH, sheet_items
from rallyiq.events.config import EventsConfig
B5=Path('~/Desktop/RallyIQ/train_v5').expanduser()
ROWS={'own_20260906_0617':['r4#9'],'own_20260906_0618':['r4#0','r4#1','r4#13'],'own_20260906_0619':['r4#14','r4b#34','r4b#19','r4b#20'],'own_20260906_0616':['S122','S130']}
its=sheet_items.items()
lovo=Path('~/Desktop/RallyIQ/train_v4/ckpt/lovo_B5').expanduser()
for c in AH.load_clips(B5/'inputs_C',B5/'labels/amateur_v4.json',only=list(ROWS)):
    lp=lovo/f'lovo_{c.cid}.json'; cfg=replace(EventsConfig(),**{**AH.AMATEUR_BASE,'hit_scorer':str(lp if lp.exists() else lovo/'full.json')})
    rl=[(r.start_ms/1000,r.end_ms/1000,len(r.hits)) for r in c.run(cfg)]
    for it in its[c.cid]:
        if it['id'] not in ROWS[c.cid]: continue
        for nm,t in (('hit',it['t']),('next',it.get('next_t'))):
            if t is None: continue
            inr=[r for r in rl if r[0]-.5<=t<=r[1]+.5]
            print(c.cid[-7:],it['id'],nm,t,'IN' if inr else 'out',inr[:1])
