"""d_pipe.py KEY... : fresh videos -> inputs/d_run/<KEY>/ with v5-code replayed players (pose dump) and the v6 tracker's
shuttle from the saved candidates (CANDS env: alternative candidates file per key, e.g. ckpt/b/cands_ft1)."""
import os,sys,json
os.environ.setdefault('RTREPO',os.path.expanduser('~/Desktop/RallyIQ/train_v6/repo'))
sys.path.insert(0,'/Users/mac/Desktop/RallyIQ/train_v5/scripts')
import rt,replay
from rallyiq.contracts import CourtCalibration,VideoMeta
B6=os.path.expanduser('~/Desktop/RallyIQ/train_v6'); out_root=os.environ.get('OUT',f'{B6}/inputs/d_run')
def lk(a,b):
    if os.path.lexists(b): os.remove(b)
    os.symlink(a,b)
for k in sys.argv[1:]:
    src=f'{B6}/inputs/d/{k}'; d=f'{out_root}/new_runs/{k}'; os.makedirs(d,exist_ok=True)
    for n in ('probe.json','calibration.json','source.mp4'): lk(f'{src}/{n}',f'{d}/{n}')
    pp=f'{B6}/inputs/d_players/{k}.json'
    if not os.path.exists(pp):
        os.makedirs(os.path.dirname(pp),exist_ok=True)
        rt.IDS[k]=[k]; rt.art=lambda kk,_s=src: _s
        by=replay.load.__wrapped__(k) if hasattr(replay.load,'__wrapped__') else None
        import numpy as np
        D=np.load(f'{B6}/dumps/{k}.npz'); byf={}
        from rallyiq.perception.players.models import PersonDetection
        hi=D['hist'].astype(np.float32)
        for i in range(len(D['frame'])): byf.setdefault(int(D['frame'][i]),[]).append((PersonDetection(bbox=D['bbox'][i].copy(),score=float(D['score'][i]),keypoints=D['kps'][i].copy()),None if np.isnan(hi[i][0]) else hi[i]))
        cal=CourtCalibration.model_validate_json(open(f'{src}/calibration.json').read())
        t,r=replay.replay(k,'auto',by=byf,cal=cal); open(pp,'w').write(t.model_dump_json()); print(k,'players',r.match_format,r.format_decision,flush=True)
    lk(pp,f'{d}/players.json'); lk(f'{src}/shuttle.json',f'{d}/shuttle.json')
    cands=os.environ.get('CANDS'); cp=f'{cands}/{k}.json' if cands else f'{src}/shuttle_candidates.json'
    c=rt.Ctx(k,d,cp); tr,_=c.run(); os.remove(f'{d}/shuttle.json'); open(f'{d}/shuttle.json','w').write(tr.model_dump_json())
    print(k,'shuttle',len(c.old.points),'->',len(tr.points),flush=True)
