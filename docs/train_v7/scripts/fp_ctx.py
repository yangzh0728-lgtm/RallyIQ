"""fp_ctx.py: context of every scorer hit in labelled windows (am7 v6 labels + fresh set): rally hit count, position, gaps,
p_swing of hitter/other side -> ckpt/fp_ctx.json + summary by label."""
import os,sys,json
os.environ['RTREPO']=os.path.expanduser('~/Desktop/RallyIQ/train_v7/repo'); sys.path.insert(0,'/Users/mac/Desktop/RallyIQ/train_v5/scripts')
import rt
from pathlib import Path
from dataclasses import replace
import amateur_hits as AH
from rallyiq.events.config import EventsConfig
from multiprocessing import Pool
B6=Path('~/Desktop/RallyIQ/train_v6').expanduser()
TR=AH.load_truth(Path('/Users/mac/Desktop/RallyIQ/train_v5/inputs_C'),Path('/Users/mac/Desktop/RallyIQ/train_v7/labels/amateur_v6.json'))
def fresh(k):
    L=[l.split('|') for l in open(B6/f'labels/d/{k}.txt') if l[0].isdigit()]
    labs=[{'t':float(x[2]),'kind':{'hit':'pos','not_a_hit':'neg','unsure_hit':'ign'}[x[1].strip()],'side':x[3].strip()} for x in L]
    return {'set':'fresh','windows':json.load(open(B6/f'ckpt/d/v6/{k}.json'))['windows'],'labels':labs}
def work(k):
    if k.startswith('own_20260529_000') and k[-1] in '579':
        art=B6/f'inputs/d_run/new_runs/{k}'; truth=fresh(k); sc='/Users/mac/Desktop/RallyIQ/train_v4/ckpt/lovo_B5/full.json'
    else:
        art=Path(f'/Users/mac/Desktop/RallyIQ/train_v5/inputs_C/new_runs/{k}'); truth=TR[k]; lp=Path(f'/Users/mac/Desktop/RallyIQ/train_v4/ckpt/lovo_B5/lovo_{k}.json'); sc=str(lp if lp.exists() else lp.parent/'full.json')
    c=AH.AmClip(k,art,truth); R=c.run(replace(EventsConfig(),**{**AH.AMATEUR_BASE,'hit_scorer':sc}))
    out=[]
    for r in R:
        for i,h in enumerate(r.hits):
            t=h.t_ms/1000
            if not c.in_window(t): continue
            m=[g for g in truth['labels'] if abs(g['t']-t)<=.2]; lab=min(m,key=lambda g:abs(g['t']-t))['kind'] if m else 'unl'
            out.append({'k':k,'t':t,'lab':lab,'n':len(r.hits),'i':i,'conf':h.confidence,'prev':(t-r.hits[i-1].t_ms/1000) if i else None,'next':(r.hits[i+1].t_ms/1000-t) if i+1<len(r.hits) else None,
                        'ps':h.scores.get('p_swing'),'src':h.source,'reasons':h.reasons})
    return out
if __name__=='__main__':
    keys=[k for k in TR if TR[k]['set']!='qa6']+['own_20260529_0005','own_20260529_0007','own_20260529_0009']
    with Pool(6) as p: rows=[x for r in p.map(work,keys) for x in r]
    json.dump(rows,open(os.path.expanduser('~/Desktop/RallyIQ/train_v7/ckpt/fp_ctx.json'),'w'))
    import collections
    for lab in ('pos','neg','unl'):
        R=[x for x in rows if x['lab']==lab]; print(lab,len(R),'single-hit rally',sum(x['n']==1 for x in R),'first',sum(x['i']==0 for x in R),'last',sum(x['i']==x['n']-1 for x in R),'conf<.5',sum(x['conf']<.5 for x in R))
