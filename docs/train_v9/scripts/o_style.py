"""o_style.py: playing-style feature stability. Per own video (full video, v7 events + strokes with the own residual): style
features per player, bootstrap over rallies (1,000) for 95% CIs, and split-half (odd vs even rallies) differences.
-> ckpt/o_style.json + printed table."""
import os,sys,json,random,math
R=os.path.expanduser('~/Desktop/RallyIQ/train_v9/repo'); sys.path[:0]=[R,R+'/scripts']
from pathlib import Path
from dataclasses import replace
import numpy as np
import amateur_hits as AH
from rallyiq.contracts import ShuttleTrack,PlayerTracks
from rallyiq.events.config import EventsConfig
from rallyiq.events.adapters import events_to_contract
from rallyiq.strokes.classify import classify,StrokesConfig,merge_strokes
from rallyiq.strokes.detail import get_model as get_attrs
from rallyiq.strokes.learned import BUNDLED,make_model_fn
from rallyiq.strokes import court as CM
from rallyiq.report import style as ST
from multiprocessing import Pool
FEATS=['smash_pct','net_pct','clear_pct','drop_pct','lift_block_pct','attack_pct','defense_pct']
B5=Path('/Users/mac/Desktop/RallyIQ/train_v5/inputs_C/new_runs'); B6=Path('/Users/mac/Desktop/RallyIQ/train_v6/inputs/d_run/new_runs')
KEYS=['own_20260302_07','own_20260310_01','own_20260320_0239','own_20260324_0241','own_20260324_0245','own_20260906_0616','own_20260906_0617',
      'own_20260906_0618','own_20260906_0619','own_20260906_0621','own_20260906_0622','own_20260529_0005','own_20260529_0007','own_20260529_0009']
def work(k):
    art=(B6 if k.startswith('own_20260529_000') else B5)/k
    c=AH.AmClip(k,art,{'set':'x','windows':[],'labels':[]}); meta=json.loads(AH.read_json(art,'probe'))
    ev=events_to_contract(meta['video_id'],c.run(replace(EventsConfig(),**AH.AMATEUR_BASE)),c.match_format)
    sh=ShuttleTrack.model_validate_json(AH.read_json(art,'shuttle')); pl=PlayerTracks.model_validate_json(AH.read_json(art,'players'))
    attrs=get_attrs(); model=CM.make_model_fn(make_model_fn(BUNDLED),CM.get_model(),attrs,c.fps,seq=True,residual=CM.get_residual(None))
    st=classify(ev,sh,pl,c.fps,StrokesConfig(),model=model,model_name='x',detail=True,attrs=attrs)
    ev=merge_strokes(ev,st); rallies=ST._contexts(ev,pl,c.fps); doubles=ev.match_format=='doubles'
    subs=sorted({x.side for r in rallies for x in r if x.side})
    out={}
    rng=random.Random(0)
    for s in subs:
        base=ST.features(rallies,s,doubles); n=base['smash_pct'][1]
        if n<20: continue
        bs={f:[] for f in FEATS}
        for _ in range(1000):
            smp=[rallies[rng.randrange(len(rallies))] for _ in rallies]; f=ST.features(smp,s,doubles)
            for q in FEATS:
                if f[q][0] is not None: bs[q].append(f[q][0])
        odd=ST.features(rallies[0::2],s,doubles); even=ST.features(rallies[1::2],s,doubles)
        out[s]={'n':n,'rallies':len(rallies),**{q:{'v':base[q][0],'lo':float(np.percentile(bs[q],2.5)),'hi':float(np.percentile(bs[q],97.5)),
                 'half':(odd[q][0] or 0)-(even[q][0] or 0)} for q in FEATS}}
    return k,ev.match_format,out
if __name__=='__main__':
    with Pool(6) as p: res=p.map(work,KEYS)
    json.dump({k:{'format':f,'players':o} for k,f,o in res},open(os.path.expanduser('~/Desktop/RallyIQ/train_v9/ckpt/o_style.json'),'w'),indent=1)
    W={q:[] for q in FEATS}
    for k,f,o in res:
        for s,d in o.items():
            print(k[-7:],f[:3],s,'shots',d['n'],'rallies',d['rallies'],' '.join(f"{q[:-4]}={d[q]['v']:.0f}[{d[q]['lo']:.0f},{d[q]['hi']:.0f}]" for q in FEATS))
            for q in FEATS: W[q].append(d[q]['hi']-d[q]['lo'])
    print('median CI width (points):',{q:round(float(np.median(v)),1) for q,v in W.items()})
