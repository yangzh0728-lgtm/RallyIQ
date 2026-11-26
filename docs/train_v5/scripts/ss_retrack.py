"""Retrack the ShuttleSet clips with the current tracker -> inputs/public_v5/<set> (other artifacts symlinked).
ss_retrack.py OWN(0|1) OUTNAME"""
import sys,os,json; sys.path.insert(0,'/Users/mac/Desktop/RallyIQ/train_v5/scripts')
import rt
from multiprocessing import Pool
own,out=sys.argv[1]!='0',sys.argv[2]
SRC=rt.H+'/train_v3/inputs/public'
def f(a):
    s,cid,e=a; ad=e['artifacts']; vid=ad.rstrip('/').split('/')[-2]
    cp=f'{rt.B}/inputs/ss_cands/{s}/{vid}.json'
    od=f'{rt.B}/inputs/{out}/{s}/data/videos/{vid}/artifacts'; os.makedirs(od,exist_ok=True)
    for n in os.listdir(ad):
        if n.startswith('.') or n.startswith('shuttle.json'): continue
        if not os.path.lexists(f'{od}/{n}'): os.symlink(f'{ad}/{n}',f'{od}/{n}')
    c=rt.Ctx('x',ad,cp); tr,_=c.run(own=own)
    open(f'{od}/shuttle.json','w').write(tr.model_dump_json())
    a={p.frame:(p.x,p.y) for p in c.old.points}; b={p.frame:(p.x,p.y) for p in tr.points}
    same=sum(1 for q in a if q in b and abs(a[q][0]-b[q][0])+abs(a[q][1]-b[q][1])<2)
    return s,cid,od,len(a),len(b),same
if __name__=='__main__':
    jobs=[];idx={}
    for s in ('shuttleset_test_v3','shuttleset_train_v3'):
        idx[s]=json.load(open(f'{SRC}/{s}/index.json')); jobs+=[(s,c,e) for c,e in idx[s].items() if 'artifacts' in e]
    with Pool(4) as p:
        for s,cid,od,a,b,same in p.imap(f,jobs):
            idx[s][cid]['artifacts']=od
            if s.endswith('test_v3'): print(cid,a,b,same,flush=True)
    for s in idx:
        d=f'{rt.B}/inputs/{out}/{s}'; json.dump(idx[s],open(d+'/index.json','w'))
        for n in ('labels','preds'):
            if not os.path.lexists(f'{d}/{n}') and os.path.exists(f'{SRC}/{s}/{n}'): os.symlink(f'{SRC}/{s}/{n}',f'{d}/{n}')
