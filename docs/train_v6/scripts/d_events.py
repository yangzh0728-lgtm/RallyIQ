"""d_events.py KEY... : events on inputs/d_run (bundled scorer, amateur settings) + rules-only candidates, inside 3 fixed
windows (40 s at 20/50/80 % of the video, chosen by position only) -> ckpt/d/<KEY>.json"""
import sys,os,json
from pathlib import Path
from dataclasses import replace
R=Path('~/Desktop/RallyIQ/train_v6/repo').expanduser(); sys.path[:0]=[str(R),str(R/'scripts')]
import amateur_hits as AH
from rallyiq.events.config import EventsConfig
B6=Path('~/Desktop/RallyIQ/train_v6').expanduser(); root=Path(os.environ.get('DRUN',B6/'inputs/d_run')); out=B6/'ckpt/d'/os.environ.get('TAG','v6'); out.mkdir(parents=True,exist_ok=True)
for k in sys.argv[1:]:
    art=root/'new_runs'/k; dur=json.load(open(art/'probe.json'))['duration_ms']/1000
    wins=[(round(dur*f,2),round(dur*f+40,2)) for f in (.2,.5,.8)]
    c=AH.AmClip(k,art,{'set':'d','windows':wins,'labels':[]})
    preds=c.predict_full(replace(EventsConfig(),**AH.AMATEUR_BASE))
    rules=c.predict_full(replace(EventsConfig(),**{**AH.AMATEUR_BASE,'hit_scorer':'off','swing_veto':0.0}))
    json.dump({'windows':wins,'preds':[p for p in preds if c.in_window(p['t'])],'rules':[p for p in rules if c.in_window(p['t'])]},open(out/f'{k}.json','w'))
    print(k,wins,len([p for p in preds if c.in_window(p['t'])]),'hits',len([p for p in rules if c.in_window(p['t'])]),'rule hits',flush=True)
