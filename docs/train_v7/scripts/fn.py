"""fn.py PREDS [INPUTS]: missed labelled hits by cause -> min wrist distance (player heights) of track points within +-0.1 s."""
import sys,os,json,numpy as np
from pathlib import Path
R=Path('~/Desktop/RallyIQ/train_v6/repo').expanduser(); sys.path[:0]=[str(R),str(R/'scripts')]
import amateur_hits as AH
from rallyiq.events.config import EventsConfig
from rallyiq.events.players import PoseIndex
B5=Path('~/Desktop/RallyIQ/train_v5').expanduser(); B6=Path('~/Desktop/RallyIQ/train_v6').expanduser()
name=sys.argv[1]; inp=Path(sys.argv[2]) if len(sys.argv)>2 else B5/'inputs_C'
preds=json.load(open(next(p for p in (B6/f'ckpt/preds/{name}.json',B5/f'ckpt/preds/{name}.json') if p.exists())))
cfg=EventsConfig(); out=[]; C={}
for c in AH.load_clips(inp,B5/'labels/amateur_v4.json'):
    idx=PoseIndex(c.poses,cfg); pts=[p for p in c.shuttle if p.visible and p.x is not None]; ts=np.array([p.t_ms for p in pts])
    for g in c.truth['labels']:
        if g['kind']!='pos' or not c.in_window(g['t']): continue
        if any(abs(p['t']-g['t'])<=.2 for p in preds[c.cid]): continue
        js=np.where(np.abs(ts-g['t']*1000)<=100)[0]
        if not len(js): cause='no_track'; d=None
        else:
            d=min((min((x.distance_rel for x in idx.proximities(pts[j].frame,(pts[j].x,pts[j].y)) if not x.filled),default=9)) for j in js)
            cause='at_wrist' if d<.35 else ('near' if d<1 else 'far_from_players')
        C[cause]=C.get(cause,0)+1; out.append({'k':c.cid,'set':c.truth['set'],'t':g['t'],'cause':cause,'d':d})
json.dump(out,open(B6/f'ckpt/fn_{name}.json','w'),indent=0)
print(C); import collections; print(collections.Counter((o['set'],o['cause']) for o in out))
