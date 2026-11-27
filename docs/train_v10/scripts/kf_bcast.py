"""kf_bcast.py: grouped 5-fold by match over all ShuttleSet clips (see PLAN.md follow-up 2)."""
import os,sys,glob,random,time,json
import numpy as np, torch
sys.argv=[sys.argv[0],'kf','--seeds','2']
B=os.path.expanduser('~/Desktop/RallyIQ/train_v10')
exec(open(f'{B}/scripts/r_train.py').read().split('t0=time.time(); P={}')[0])
R=os.path.expanduser('~/Desktop/RallyIQ/train_v10/repo'); sys.path[:0]=[R,R+'/scripts']
from pathlib import Path
import learn_hits as LH
from evaluation import metrics as MM
from rallyiq.events.config import EventsConfig
sys.path.insert(0,R); from importlib import util
import rallyiq.events; spec=util.spec_from_file_location('rallyiq.events.seqhits',f'{B}/ckpt/unshipped_code/seqhits.py'); SH=util.module_from_spec(spec); SH.__package__='rallyiq.events'; spec.loader.exec_module(SH)
a_=json.load(open('/Users/mac/Desktop/RallyIQ/train_v5/inputs/pub_v5/shuttleset_train_v3/index.json')); b_=json.load(open('/Users/mac/Desktop/RallyIQ/train_v5/inputs/pub_v5/shuttleset_test_v3/index.json'))
match={**{'sstrain__'+k:v['match_id'] for k,v in a_.items() if 'match_id' in v},**{'sstest__'+k:v['match_id'] for k,v in b_.items() if 'match_id' in v}}
keys=[k for k in match if k in D]; M=sorted(set(match[k] for k in keys),key=str); fold={m:i%5 for i,m in enumerate(M)}
clips={**{'sstrain__'+c.cid:c for c in LH.load_clips(Path('/Users/mac/Desktop/RallyIQ/train_v5/inputs/pub_v5/shuttleset_train_v3'),'index.json')},
       **{'sstest__'+c.cid:c for c in LH.load_clips(Path('/Users/mac/Desktop/RallyIQ/train_v5/inputs/pub_v5/shuttleset_test_v3'),'results.json')}}
keys=[k for k in keys if k in clips]
spans={k:[(r.start_ms,r.end_ms) for r in clips[k].run(EventsConfig())] for k in keys}
def side_of(c):
    pf={f.frame:{p.player_id:p.side for p in f.players} for f in c.poses}
    def s(h):
        for dd in range(12):
            for f in (h.frame-dd,h.frame+dd):
                if f in pf and h.hitter_id in pf[f]: return pf[f][h.hitter_id]
    return s
rules={k:[{'t_ms':clips[k].offset+h.t_ms,'side':side_of(clips[k])(h),'ok':True} for r in clips[k].run(EventsConfig()) for h in r.hits] for k in keys}
def seq_preds(k,h,far,thr,gap):
    d=D[k]; fps=float(d['fps']); g=d['grid']; out=[]
    for i in SH.peaks(g,h,fps,thr,gap):
        t=g[i]*1000/fps
        if any(a-300<=t<=b+300 for a,b in spans[k]): out.append({'t_ms':clips[k].offset+t,'side':'far' if far[i]>.5 else 'near','ok':True})
    return out
def counts(k,hs): st=MM.eval_hits(clips[k].label,{'hits':hs}); return np.array([st.tp,st.fp,st.fn])
def F(c): return 2*c[0]/max(2*c[0]+c[1]+c[2],1)
def train_predict(tr,te,seeds):
    global mu,sd
    mu=np.concatenate([D[k]['X'] for k in tr]).mean(0); sd=np.concatenate([D[k]['X'] for k in tr]).std(0)+1e-3
    P={}
    for s in range(seeds):
        torch.manual_seed(s); m=TCN(27,64).to(dev); train(m,tr,[1]*len(tr),3000,s)
        for k in te: h,f=predict(m,k); P.setdefault(k,[]).append((h,f))
    return {k:(np.mean([a for a,_ in v],0),np.mean([b for _,b in v],0)) for k,v in P.items()}
GRID=[(t,g) for t in (0.2,0.3,0.4,0.5,0.6) for g in (0.2,0.28,0.36)]
res={}; t0=time.time()
for f in range(5):
    te=[k for k in keys if fold[match[k]]==f]; tr=[k for k in keys if fold[match[k]]!=f]
    trM=sorted({match[k] for k in tr},key=str); inner={m:i%3 for i,m in enumerate(trM)}; oof={}
    for j in range(3):
        itr=[k for k in tr if inner[match[k]]!=j]; ite=[k for k in tr if inner[match[k]]==j]; oof.update(train_predict(itr,ite,1))
    best=max(GRID,key=lambda tg:F(sum(counts(k,seq_preds(k,*oof[k],*tg)) for k in tr)))
    P=train_predict(tr,te,2)
    for k in te: res[k]={'seq':counts(k,seq_preds(k,*P[k],*best)).tolist(),'rules':counts(k,rules[k]).tolist(),'match':str(match[k]),'thr':best}
    print('fold',f,'chosen',best,'test clips',len(te),round(time.time()-t0),flush=True)
json.dump(res,open(f'{B}/ckpt/kf_bcast.json','w'))
mm=sorted({v['match'] for v in res.values()})
def Fm(nm,ms): return F(sum(np.array(v[nm]) for v in res.values() if v['match'] in ms))
rng=random.Random(0); ds=sorted(Fm('seq',s)-Fm('rules',s) for s in ([rng.choice(mm) for _ in mm] for _ in range(5000)))
print('matches',len(mm),'rules %.4f seq %.4f diff %+.4f CI [%+.4f, %+.4f]'%(Fm('rules',mm),Fm('seq',mm),Fm('seq',mm)-Fm('rules',mm),ds[125],ds[4875]))
