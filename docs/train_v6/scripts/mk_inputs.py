"""mk_inputs.py NAME CANDDIR: inputs_NAME = train_v5/inputs_C (v5 players, calibration, sources) with shuttle.json rebuilt
by the v6 tracker from CANDDIR/<key>.json. Other root files symlinked like inputs_C."""
import os,sys
os.environ.setdefault('RTREPO',os.path.expanduser('~/Desktop/RallyIQ/train_v6/repo'))
sys.path.insert(0,'/Users/mac/Desktop/RallyIQ/train_v5/scripts')
import rt
from multiprocessing import Pool
name,cd=sys.argv[1],sys.argv[2]; B6=os.path.expanduser('~/Desktop/RallyIQ/train_v6'); C=rt.B+'/inputs_C'; R=f'{B6}/inputs_{name}'
def f(k):
    s=f'{C}/new_runs/{k}'; d=f'{R}/new_runs/{k}'; os.makedirs(d,exist_ok=True)
    for n in os.listdir(s):
        if n.startswith('.') or n.startswith('shuttle'): continue
        if not os.path.lexists(f'{d}/{n}'): os.symlink(os.path.realpath(f'{s}/{n}'),f'{d}/{n}')
    c=rt.Ctx(k,s,f'{cd}/{k}.json'); tr,_=c.run(); open(f'{d}/shuttle.json','w').write(tr.model_dump_json()); return k,len(c.old.points),len(tr.points)
if __name__=='__main__':
    os.makedirs(f'{R}/new_runs',exist_ok=True)
    for n in os.listdir(C):
        if n in('new_runs',) or n.startswith('.'): continue
        if not os.path.lexists(f'{R}/{n}'): os.symlink(os.path.realpath(f'{C}/{n}'),f'{R}/{n}')
    with Pool(4) as p:
        for r in p.imap(f,sorted(rt.IDS)): print(*r,flush=True)
