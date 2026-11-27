"""k_fresh.py: stroke accuracy on the fresh-set hits I labelled (../train_v8/labels/fresh_strokes.txt) for court_v2 (+video GBT)
with and without the own-footage residual. Shots matched to labelled hits within 0.3 s."""
import os,sys,json,math
R=os.path.expanduser('~/Desktop/RallyIQ/train_v9/repo'); sys.path[:0]=[R,R+'/scripts']
from pathlib import Path
from dataclasses import replace
import amateur_hits as AH
from rallyiq.contracts import ShuttleTrack,PlayerTracks
from rallyiq.events.config import EventsConfig
from rallyiq.events.adapters import events_to_contract
from rallyiq.strokes.classify import classify,StrokesConfig
from rallyiq.strokes.detail import get_model as get_attrs
from rallyiq.strokes.learned import BUNDLED,make_model_fn
from rallyiq.strokes import court as CM
B6=Path('~/Desktop/RallyIQ/train_v6').expanduser()
items=json.load(open(os.path.expanduser('~/Desktop/RallyIQ/train_v9/ckpt/stroke_items.json')))
lab={}
for l in open(os.path.expanduser('~/Desktop/RallyIQ/train_v9/../train_v8/labels/fresh_strokes.txt')):
    if l[0].isdigit():
        n,s,sure,_=[x.strip() for x in l.split('|',3)]; lab[int(n)]=(s,sure=='true')
attrs=get_attrs(); court=CM.get_model(); res=CM.get_residual(None); res9=CM.get_residual(os.path.expanduser('~/Desktop/RallyIQ/train_v9/ckpt/resid_v9.json'))
res_out={'base':[],'resid':[],'resid_v9':[]}
for k in ('own_20260529_0005','own_20260529_0007','own_20260529_0009'):
    art=B6/f'inputs/d_run/new_runs/{k}'; c=AH.AmClip(k,art,{'set':'d','windows':[],'labels':[]})
    meta=json.loads(AH.read_json(art,'probe')); R_=c.run(replace(EventsConfig(),**AH.AMATEUR_BASE))
    ev=events_to_contract(meta['video_id'],R_,c.match_format)
    sh=ShuttleTrack.model_validate_json(AH.read_json(art,'shuttle')); pl=PlayerTracks.model_validate_json(AH.read_json(art,'players'))
    for nm,rz in (('base',None),('resid',res),('resid_v9',res9)):
        model=CM.make_model_fn(make_model_fn(BUNDLED),court,attrs,c.fps,seq=True,residual=rz)
        out=classify(ev,sh,pl,c.fps,StrokesConfig(),model=model,model_name=nm,detail=True,attrs=attrs)
        tm={s.shot_id:s.t_ms/1000 for s in ev.shots}
        for i,it in enumerate(items):
            if it['k']!=k or lab[i][0]=='-': continue
            m=[s for s in out.shots if abs(tm[s.shot_id]-it['t'])<=.3]
            pred=min(m,key=lambda s:abs(tm[s.shot_id]-it['t'])).stroke.shot_type if m else None
            res_out[nm].append((k,i,lab[i][0],lab[i][1],pred))
json.dump(res_out,open(os.path.expanduser('~/Desktop/RallyIQ/train_v9/ckpt/k_fresh.json'),'w'))
for nm,rows in res_out.items():
    m=[r for r in rows if r[4]]; sure=[r for r in m if r[3]]
    print(nm,'matched',len(m),'of',len(rows),'acc all %.3f'%(sum(r[2]==r[4] for r in m)/len(m)),'acc sure %.3f (%d)'%(sum(r[2]==r[4] for r in sure)/max(len(sure),1),len(sure)))
