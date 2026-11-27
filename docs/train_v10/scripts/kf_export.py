"""kf_export.py: (1) production-path check: per outer fold, export the fold's TCN as a seq_hits.npz and run the real
segment_events(hit_model=<path>) on the held-out clips, next to the harness decode; (2) choose thr/gap on the pooled
5-fold OOF probabilities; (3) train 3 seeds on all matches and export repo/rallyiq/events/models/seq_hits.npz."""
import os,sys,json,random,time
from dataclasses import replace
B=os.path.expanduser('~/Desktop/RallyIQ/train_v10')
src=open(f'{B}/scripts/kf_bcast.py').read().split('res={}; t0=time.time()')[0]
exec(src)
def export(models,thr,gap,path):
    out={'mu':mu,'sd':sd,'thr':np.array(thr),'gap_s':np.array(gap),'n_members':np.array(len(models))}
    for s,m in enumerate(models):
        for k,v in m.state_dict().items(): out[f'm{s}/{k}']=v.cpu().numpy().astype(np.float32)
    np.savez_compressed(path,**out)
def fit(tr,seeds):
    global mu,sd
    mu=np.concatenate([D[k]['X'] for k in tr]).mean(0); sd=np.concatenate([D[k]['X'] for k in tr]).std(0)+1e-3
    ms=[]
    for s in range(seeds):
        torch.manual_seed(s); m=TCN(27,64).to(dev); train(m,tr,[1]*len(tr),3000,s); m.eval(); ms.append(m)
    return ms
def preds(ms,te):
    out={}
    for k in te:
        hs=[predict(m,k) for m in ms]; out[k]=(np.mean([a for a,_ in hs],0),np.mean([b for _,b in hs],0))
    return out
chosen=[(0.4,0.28),(0.2,0.36),(0.5,0.36),(0.3,0.28),(0.3,0.28)]
oof={}; chk={}; os.makedirs(f'{B}/ckpt/kf',exist_ok=True)
for f in range(5):
    te=[k for k in keys if fold[match[k]]==f]; tr=[k for k in keys if fold[match[k]]!=f]
    ms=fit(tr,2); P=preds(ms,te); oof.update(P)
    path=f'{B}/ckpt/kf/fold{f}.npz'; export(ms,*chosen[f],path)
    cfg=replace(EventsConfig(),hit_model=path)
    for k in te:
        c=clips[k]; hs=[{'t_ms':c.offset+h.t_ms,'side':side_of(c)(h),'ok':True} for r in c.run(cfg) for h in r.hits]
        chk[k]={'harness':counts(k,seq_preds(k,*P[k],*chosen[f])).tolist(),'prod':counts(k,hs).tolist(),'rules':counts(k,rules[k]).tolist(),'match':str(match[k])}
    print('fold',f,'done',flush=True)
json.dump(chk,open(f'{B}/ckpt/kf_prodcheck.json','w'))
for nm in ('rules','harness','prod'): print(nm,round(F(sum(np.array(v[nm]) for v in chk.values())),4))
mm=sorted({v['match'] for v in chk.values()})
def Fm(nm,ms): return F(sum(np.array(v[nm]) for v in chk.values() if v['match'] in ms))
rng=random.Random(0); ds=sorted(Fm('prod',s)-Fm('rules',s) for s in ([rng.choice(mm) for _ in mm] for _ in range(5000)))
print('prod vs rules diff %+.4f CI [%+.4f, %+.4f]'%(Fm('prod',mm)-Fm('rules',mm),ds[125],ds[4875]))
best=max(GRID,key=lambda tg:F(sum(counts(k,seq_preds(k,*oof[k],*tg)) for k in keys)))
print('final thr/gap on pooled OOF',best,flush=True)
ms=fit(keys,3); export(ms,*best,f'{R}/rallyiq/events/models/seq_hits.npz'); print('exported',os.path.getsize(f'{R}/rallyiq/events/models/seq_hits.npz'))
