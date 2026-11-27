"""kf_sidecheck.py: where does the harness-vs-production gap come from? time-only (any side) vs same-side F1."""
import os,sys,json
from dataclasses import replace
B=os.path.expanduser('~/Desktop/RallyIQ/train_v10')
exec(open(f'{B}/scripts/kf_bcast.py').read().split('res={}; t0=time.time()')[0])
chosen=[(0.4,0.28),(0.2,0.36),(0.5,0.36),(0.3,0.28),(0.3,0.28)]
from rallyiq.events import seqhits as SHr
T={n:np.zeros(4) for n in ('rules','harness','prod')}
def acc(n,k,hs):
    st=MM.eval_hits(clips[k].label,{'hits':hs}); T[n]+=[st.tp,st.fp+st.fn,st.tp_any_side,len(hs)]
for f in range(5):
    M=f'{B}/ckpt/kf/fold{f}.npz'; m=SHr._load(M); cfg=replace(EventsConfig(),hit_model=M)
    for k in keys:
        if fold[match[k]]!=f: continue
        c=clips[k]; h,far=m.predict(D[k]['X'])
        acc('harness',k,seq_preds(k,h,far,*chosen[f])); acc('rules',k,rules[k])
        acc('prod',k,[{'t_ms':c.offset+x.t_ms,'side':side_of(c)(x),'ok':True} for r in c.run(cfg) for x in r.hits])
ng=sum(len([g for g in clips[k].label['hits']]) for k in keys)
for n,v in T.items():
    tp,err,tpa,npred=v; print(n,'F1 same-side %.4f'%(2*tp/(2*tp+err)),'tp',int(tp),'tp_any',int(tpa),'pred',int(npred))
