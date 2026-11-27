"""kf_prodcheck.py: rerun the production-path check with the saved per-fold models (ckpt/kf/fold*.npz)."""
import os,sys,json,random
from dataclasses import replace
B=os.path.expanduser('~/Desktop/RallyIQ/train_v10')
exec(open(f'{B}/scripts/kf_bcast.py').read().split('res={}; t0=time.time()')[0])
old=json.load(open(f'{B}/ckpt/kf_prodcheck.json')); chk={}
for f in range(5):
    cfg=replace(EventsConfig(),hit_model=f'{B}/ckpt/kf/fold{f}.npz')
    for k in keys:
        if fold[match[k]]!=f: continue
        c=clips[k]; hs=[{'t_ms':c.offset+h.t_ms,'side':side_of(c)(h),'ok':True} for r in c.run(cfg) for h in r.hits]
        chk[k]=dict(old[k],prod=counts(k,hs).tolist())
json.dump(chk,open(f'{B}/ckpt/kf_prodcheck2.json','w'))
for nm in ('rules','harness','prod'): print(nm,round(F(sum(np.array(v[nm]) for v in chk.values())),4))
mm=sorted({v['match'] for v in chk.values()})
def Fm(nm,ms): return F(sum(np.array(v[nm]) for v in chk.values() if v['match'] in ms))
rng=random.Random(0); ds=sorted(Fm('prod',s)-Fm('rules',s) for s in ([rng.choice(mm) for _ in mm] for _ in range(5000)))
print('prod vs rules diff %+.4f CI [%+.4f, %+.4f] matches %d'%(Fm('prod',mm)-Fm('rules',mm),ds[125],ds[4875],len(mm)))
