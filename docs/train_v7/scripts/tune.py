"""tune.py CONFIGS.json OUT.json [--cands DIR] [--lovo DIR]: per video x config hit counts (TP/FP/FN) on amateur_v6.
CONFIGS = [{"name":..,"peak":0.3,"tracker":{...},"events":{...}}]. One worker per video keeps its context in memory;
candidates are filtered to score >= peak (production drops peaks below the checkpoint threshold)."""
import os,sys,json,argparse,time
os.environ.setdefault('RTREPO',os.path.expanduser('~/Desktop/RallyIQ/train_v7/repo'))
sys.path.insert(0,'/Users/mac/Desktop/RallyIQ/train_v5/scripts')
import rt
from pathlib import Path
from dataclasses import replace
from multiprocessing import Pool
import amateur_hits as AH
from rallyiq.events.config import EventsConfig
from rallyiq.events.adapters import shuttle_from_contract
B7=Path('~/Desktop/RallyIQ/train_v7').expanduser(); INP=Path('~/Desktop/RallyIQ/train_v5/inputs_C').expanduser()
ap=argparse.ArgumentParser(); ap.add_argument('configs'); ap.add_argument('out'); ap.add_argument('--cands',default=str(B7/'ckpt/cands/ft1'))
ap.add_argument('--lovo',default='~/Desktop/RallyIQ/train_v4/ckpt/lovo_B5'); ap.add_argument('--labels',default=str(B7/'labels/amateur_v6.json')); ap.add_argument('--keys')
a=ap.parse_args(); CFG=json.load(open(a.configs)); LOVO=Path(a.lovo).expanduser()
TRUTH=AH.load_truth(INP,Path(a.labels))
def work(k):
    t0=time.time(); art=INP/'new_runs'/k
    c=rt.Ctx(k,str(art),f'{a.cands}/{k}.json'); CS={a.cands:c.cands}
    from rallyiq.perception.shuttle import stage as S
    clip=AH.AmClip(k,art,TRUTH[k]); lp=LOVO/f'lovo_{k}.json'; sc=str(lp if lp.exists() else LOVO/'full.json')
    res={}; cache={}
    for cf in CFG:
        cd=os.path.expanduser(cf.get('cands',a.cands))
        if cd not in CS: CS[cd]=S.load_candidates(Path(cd)/f'{k}.json')[0]
        tkey=json.dumps([cd,cf.get('peak',.3),cf.get('tracker',{})],sort_keys=True)
        if tkey not in cache:
            p=cf.get('peak',.3); c.cands={f:[x for x in v if x[2]>=p] for f,v in CS[cd].items()}
            tr,_=c.run(peak=p,**cf.get('tracker',{})); cache[tkey]=shuttle_from_contract(tr)
        clip.shuttle=cache[tkey]
        ev=replace(EventsConfig(),**{**AH.AMATEUR_BASE,'hit_scorer':cf.get('scorer',sc) if cf.get('scorer')!='lovo' else sc,**cf.get('events',{})})
        if 'scorer_dir' in cf:
            d=Path(cf['scorer_dir']).expanduser(); q=d/f'lovo_{k}.json'; ev=replace(ev,hit_scorer=str(q if q.exists() else d/'full.json'))
        if 'dead_dir' in cf:
            d=Path(cf['dead_dir']).expanduser(); q=d/f'lovo_{k}.json'; ev=replace(ev,dead_ball=str(q if q.exists() else d/'full.json'))
        s=AH.score_clip(clip,clip.predict_full(ev)); res[cf['name']]={x:s[x] for x in ('tp','fp','fn')}
    return k,res,round(time.time()-t0)
if __name__=='__main__':
    keys=a.keys.split(',') if a.keys else sorted(TRUTH)
    out={}
    with Pool(int(os.environ.get('NP',5))) as p:
        for k,r,dt in p.imap_unordered(work,keys): out[k]=r; print(k,dt,'s',flush=True)
    json.dump(out,open(a.out,'w'))
    QA={'own_20260529_0003','own_20260529_0004','own_20260529_0006','own_20260529_0008','own_20260906_0615','own_20260906_0620'}
    for cf in CFG:
        n=cf['name']; f=lambda ks:(lambda t:2*t[0]/max(2*t[0]+t[1]+t[2],1))([sum(out[k][n][x] for k in ks) for x in ('tp','fp','fn')])
        print(f"{n:40s} all {f(keys):.4f} other {f([k for k in keys if k not in QA]):.4f} qa6 {f([k for k in keys if k in QA]):.4f}")
