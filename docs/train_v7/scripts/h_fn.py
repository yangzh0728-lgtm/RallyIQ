"""h_fn.py: every labelled hit the shipped pipeline misses (am7 + qa6 + fresh), with what the candidate stage saw near it:
nearest candidate (dt, rule conf, scorer prob, reasons) and the nearest kept hits before/after -> ckpt/h_fn.json"""
import os,sys,json
os.environ['RTREPO']=os.path.expanduser('~/Desktop/RallyIQ/train_v7/repo'); sys.path.insert(0,'/Users/mac/Desktop/RallyIQ/train_v5/scripts')
import rt
from pathlib import Path
from dataclasses import replace
import amateur_hits as AH
from rallyiq.events.config import EventsConfig
from rallyiq.events.pipeline import segment_events
from multiprocessing import Pool
sys.path.insert(0,os.path.dirname(__file__)); import g_feats as G
def work(k):
    if k in G.FRESH: art=G.B6/f'inputs/d_run/new_runs/{k}'; truth=G.fresh(k); sc='/Users/mac/Desktop/RallyIQ/train_v4/ckpt/lovo_B5/full.json'
    else:
        art=Path(f'/Users/mac/Desktop/RallyIQ/train_v5/inputs_C/new_runs/{k}'); truth=G.TR[k]; lp=Path(f'/Users/mac/Desktop/RallyIQ/train_v4/ckpt/lovo_B5/lovo_{k}.json'); sc=str(lp if lp.exists() else lp.parent/'full.json')
    c=AH.AmClip(k,art,truth); cfg=replace(EventsConfig(),**{**AH.AMATEUR_BASE,'hit_scorer':sc}); sink=[]
    R=segment_events(c.shuttle,c.poses,cfg,video_end_ms=c.duration_ms,collect=sink,off_court=c.off_court(cfg))
    hits=sorted(h.t_ms/1000 for r in R for h in r.hits); cands=[(pt.t_ms/1000,cd.confidence,list(cd.reasons)) for _,_,cd,pt in sink]
    out=[]
    for g in truth['labels']:
        if g['kind']!='pos' or not c.in_window(g['t']) or any(abs(h-g['t'])<=.2 for h in hits): continue
        near=sorted(cands,key=lambda x:abs(x[0]-g['t']))[:1]
        prev=max([h for h in hits if h<g['t']],default=None); nxt=min([h for h in hits if h>g['t']],default=None)
        out.append({'k':k,'set':truth['set'],'t':g['t'],'side':g.get('side'),'cand':near[0] if near and abs(near[0][0]-g['t'])<=.25 else None,
                    'gap_prev':None if prev is None else round(g['t']-prev,2),'gap_next':None if nxt is None else round(nxt-g['t'],2)})
    return out
if __name__=='__main__':
    keys=sorted(G.TR)+list(G.FRESH)
    with Pool(6) as p: R=[x for r in p.map(work,keys) for x in r]
    json.dump(R,open(G.B7/'ckpt/h_fn.json','w'))
    import collections
    print(len(R),collections.Counter(x['set'] for x in R))
    print('no candidate within 0.25 s',sum(x['cand'] is None for x in R))
    print('cand reasons',collections.Counter(r for x in R if x['cand'] for r in (x['cand'][2] or ['none'])).most_common(10))
    print('cand conf',sorted(round(x['cand'][1],2) for x in R if x['cand']))
    print('gap to nearest kept hit <0.5s',sum(1 for x in R if min(x['gap_prev'] or 9,x['gap_next'] or 9)<.5))
