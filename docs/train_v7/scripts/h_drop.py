"""h_drop.py: for the missed hits whose candidate had scorer conf >= 0.3, which stage removed them: rerun with the swing
veto off, and record whether the hit appears."""
import os,sys,json
os.environ['RTREPO']=os.path.expanduser('~/Desktop/RallyIQ/train_v7/repo'); sys.path.insert(0,'/Users/mac/Desktop/RallyIQ/train_v5/scripts')
import rt
from pathlib import Path
from dataclasses import replace
import amateur_hits as AH
from rallyiq.events.config import EventsConfig
sys.path.insert(0,os.path.dirname(__file__)); import g_feats as G
from collections import defaultdict,Counter
R=json.load(open(G.B7/'ckpt/h_fn.json')); by=defaultdict(list)
for x in R:
    if x['cand'] and x['cand'][1]>=.3: by[x['k']].append(x)
C=Counter()
for k,xs in by.items():
    if k in G.FRESH: art=G.B6/f'inputs/d_run/new_runs/{k}'; truth=G.fresh(k); sc='/Users/mac/Desktop/RallyIQ/train_v4/ckpt/lovo_B5/full.json'
    else:
        art=Path(f'/Users/mac/Desktop/RallyIQ/train_v5/inputs_C/new_runs/{k}'); truth=G.TR[k]; lp=Path(f'/Users/mac/Desktop/RallyIQ/train_v4/ckpt/lovo_B5/lovo_{k}.json'); sc=str(lp if lp.exists() else lp.parent/'full.json')
    c=AH.AmClip(k,art,truth)
    for nm,ov in (('noveto',{'swing_veto':0.0}),('nogap',{'min_hit_gap_ms':150.0}),('both',{'swing_veto':0.0,'min_hit_gap_ms':150.0})):
        P=c.predict_full(replace(EventsConfig(),**{**AH.AMATEUR_BASE,'hit_scorer':sc,**ov}))
        for x in xs: C[(nm,any(abs(p['t']-x['t'])<=.2 for p in P))]+=1
print(C)
