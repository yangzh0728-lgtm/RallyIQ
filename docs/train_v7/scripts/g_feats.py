"""g_feats.py: rally-context features of every predicted hit (shipped pipeline) in the labelled windows of the 17 own videos
(amateur_v6) and the fresh set -> ckpt/g_rows.json, plus per-video baseline TP/FP/FN (ckpt/g_base.json)."""
import os,sys,json,math
os.environ['RTREPO']=os.path.expanduser('~/Desktop/RallyIQ/train_v7/repo'); sys.path.insert(0,'/Users/mac/Desktop/RallyIQ/train_v5/scripts')
import rt
import numpy as np
from pathlib import Path
from dataclasses import replace
import amateur_hits as AH
from rallyiq.events.config import EventsConfig
from rallyiq.events import pose_hits as PH
from multiprocessing import Pool
B6=Path('~/Desktop/RallyIQ/train_v6').expanduser(); B7=Path('~/Desktop/RallyIQ/train_v7').expanduser()
TR=AH.load_truth(Path('/Users/mac/Desktop/RallyIQ/train_v5/inputs_C'),B7/'labels/amateur_v6.json')
FRESH=('own_20260529_0005','own_20260529_0007','own_20260529_0009')
def fresh(k):
    L=[l.split('|') for l in open(B6/f'labels/d/{k}.txt') if l[0].isdigit()]
    return {'set':'fresh','windows':json.load(open(B6/f'ckpt/d/v6/{k}.json'))['windows'],
            'labels':[{'t':float(x[2]),'kind':{'hit':'pos','not_a_hit':'neg','unsure_hit':'ign'}[x[1].strip()],'side':x[3].strip()} for x in L]}
MODEL=PH.get_model('auto'); OUT=os.environ.get('OUTSIDE')=='1'
def work(k):
    if k in FRESH: art=B6/f'inputs/d_run/new_runs/{k}'; truth=fresh(k); sc='/Users/mac/Desktop/RallyIQ/train_v4/ckpt/lovo_B5/full.json'
    else:
        art=Path(f'/Users/mac/Desktop/RallyIQ/train_v5/inputs_C/new_runs/{k}'); truth=TR[k]; lp=Path(f'/Users/mac/Desktop/RallyIQ/train_v4/ckpt/lovo_B5/lovo_{k}.json'); sc=str(lp if lp.exists() else lp.parent/'full.json')
    c=AH.AmClip(k,art,truth); R=c.run(replace(EventsConfig(),**{**AH.AMATEUR_BASE,'hit_scorer':sc}))
    preds=[]
    for r in R:
        for h in r.hits: preds.append({'t':h.t_ms/1000,'hitter':h.hitter_id,'side':c.side_at(h.hitter_id,h.frame),'conf':h.confidence})
    base=AH.score_clip(c,preds)
    ser=PH.side_series(c.poses); fms=1000/c.fps
    rows=[]
    for r in R:
        n=len(r.hits); sides=[c.side_at(h.hitter_id,h.frame) for h in r.hits]
        for i,h in enumerate(r.hits):
            t=h.t_ms/1000
            if (not OUT and not c.in_window(t)) or (OUT and c.in_window(t,1.0)): continue
            m=[g for g in truth['labels'] if abs(g['t']-t)<=.2]; lab=min(m,key=lambda g:abs(g['t']-t))['kind'] if m else 'unl'
            s=sides[i]; o={'near':'far','far':'near'}.get(s)
            ps,wv=PH.swing_evidence(ser,h.t_ms,s,MODEL,250.,fms) if s else (0.,0.)
            po,wo=PH.swing_evidence(ser,h.t_ms,o,MODEL,600.,fms) if o else (0.,0.)
            gp=t-r.hits[i-1].t_ms/1000 if i else 5.; gn=r.hits[i+1].t_ms/1000-t if i+1<n else 5.
            rows.append({'k':k,'set':truth['set'],'t':round(t,3),'lab':lab,'f':{
                'conf':h.confidence,'logn':math.log1p(n),'first':float(i==0),'last':float(i==n-1),'single':float(n==1),
                'gap_prev':min(gp,5.),'gap_next':min(gn,5.),'same_prev':float(i>0 and sides[i-1]==s),'same_next':float(i+1<n and sides[i+1]==s),
                'ps':ps,'wv':min(wv,20.),'ps_other':po,'wv_other':min(wo,20.),'dur':min((r.end_ms-r.start_ms)/1000,30.),
                't_start':min(t-r.start_ms/1000,10.),'t_end':min(r.end_ms/1000-t,10.)}})
    return k,rows,{x:base[x] for x in ('tp','fp','fn')}
if __name__=='__main__':
    keys=(sorted(k for k in TR if TR[k]['set']=='am7') if OUT else sorted(TR)+list(FRESH))
    with Pool(6) as p: res=p.map(work,keys)
    if OUT: json.dump([x for _,r,_ in res for x in r],open(B7/'ckpt/g_rows_out.json','w'))
    else: json.dump([x for _,r,_ in res for x in r],open(B7/'ckpt/g_rows.json','w')); json.dump({k:b for k,_,b in res},open(B7/'ckpt/g_base.json','w'))
    import collections; print(collections.Counter((x['set'],x['lab']) for _,r,_ in res for x in r))
