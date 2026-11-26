"""build_inputs.py NAME [players_dir]: inputs_NAME/new_runs/<key>/ with the retracked shuttle (and replayed players from
players_dir/<key>.json, corrected calibration) + symlinks to the rest."""
import sys; sys.path.insert(0,'/Users/mac/Desktop/RallyIQ/train_v5/scripts')
import rt,os,shutil,replay
from multiprocessing import Pool
name=sys.argv[1]; pd=sys.argv[2] if len(sys.argv)>2 else None
R=f'{rt.B}/inputs_{name}'; B4=rt.H+'/train_v4/inputs'
def lk(a,b):
    if os.path.lexists(b): os.remove(b)
    os.symlink(a,b)
def f(k):
    a=rt.art(k); d=f'{R}/new_runs/{k}'; shutil.rmtree(d,ignore_errors=True); os.makedirs(d)
    for n in os.listdir(a):
        if n.startswith('.') or n.split('.')[0] in ('shuttle','players','calibration','events','strokes','calibration_manual'): continue
        lk(f'{a}/{n}',f'{d}/{n}')
    for s in (f'{rt.B}/inputs/sources/{k}.mp4',f'{B4}/new_runs/{k}/source.mp4',f'{B4}/sources/{k}.mp4'):
        if os.path.exists(s): lk(s,f'{d}/source.mp4'); break
    cal=replay.calib(k) if pd else None
    if cal is not None: open(f'{d}/calibration.json','w').write(cal.model_dump_json())
    else:
        for e in ('.json','.json.gz'):
            if os.path.exists(f'{a}/calibration{e}'): lk(f'{a}/calibration{e}',f'{d}/calibration{e}')
    pp=f'{pd}/{k}.json' if pd else None
    if pp and os.path.exists(pp): lk(os.path.abspath(pp),f'{d}/players.json')
    else:
        for e in ('.json','.json.gz'):
            if os.path.exists(f'{a}/players{e}'): lk(f'{a}/players{e}',f'{d}/players{e}')
    c=rt.Ctx(k,d,f'{rt.B}/inputs/cands/{k}.json') if False else None
    # Ctx needs a shuttle.json for .old
    for e in ('.json','.json.gz'):
        if os.path.exists(f'{a}/shuttle{e}'): lk(f'{a}/shuttle{e}',f'{d}/shuttle{e}')
    c=rt.Ctx(k,d,f'{rt.B}/inputs/cands/{k}.json'); tr,_=c.run()
    for e in ('.json','.json.gz'):
        if os.path.lexists(f'{d}/shuttle{e}'): os.remove(f'{d}/shuttle{e}')
    open(f'{d}/shuttle.json','w').write(tr.model_dump_json()); return k,len(c.old.points),len(tr.points)
if __name__=='__main__':
    os.makedirs(R,exist_ok=True)
    for n in os.listdir(B4):
        if n in('new_runs','sources') or n.startswith('.'): continue
        if not os.path.lexists(f'{R}/{n}'): os.symlink(f'{B4}/{n}',f'{R}/{n}')
    with Pool(5) as p:
        for r in p.imap(f,rt.IDS): print(*r,flush=True)
