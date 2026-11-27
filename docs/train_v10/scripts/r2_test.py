"""r2_test.py: integrated pipeline on the 14 ShuttleSet test clips: rules (v7 default) vs hit_model=seq; per-clip counts with
LH.score (+-3 frames, same side); per-match bootstrap; numpy-vs-torch check; runtime."""
import os,sys,json,random,time
import numpy as np
R=os.path.expanduser('~/Desktop/RallyIQ/train_v10/repo'); sys.path[:0]=[R,R+'/scripts']
from pathlib import Path
from dataclasses import replace
import learn_hits as LH
from rallyiq.events.config import EventsConfig
from rallyiq.events import seqhits as SH
clips=LH.load_clips(Path('/Users/mac/Desktop/RallyIQ/train_v5/inputs/pub_v5/shuttleset_test_v3'),'results.json')
idx=json.load(open('/Users/mac/Desktop/RallyIQ/train_v5/inputs/pub_v5/shuttleset_test_v3/index.json'))
B=os.path.expanduser('~/Desktop/RallyIQ/train_v10')
# numpy vs torch ensemble probabilities
m=SH.get_model('seq'); d=np.load(f'{B}/ckpt/seq/sstest__{clips[0].cid}.npz'); h,_=m.predict(d['X']); ref=np.load(f'{B}/ckpt/r_bcast/probs.npz')[f'sstest__{clips[0].cid}__hit']
print('numpy vs torch max abs diff',float(np.abs(h-ref).max()))
def preds(c,cfg):
    pf={f.frame:{p.player_id:p.side for p in f.players} for f in c.poses}
    def side(h):
        for dd in range(0,12):
            for f in (h.frame-dd,h.frame+dd):
                if f in pf and h.hitter_id in pf[f]: return pf[f][h.hitter_id]
        return None
    return [{'t_ms':c.offset+h.t_ms,'side':side(h),'ok':True} for r in c.run(cfg) for h in r.hits]
res={}; tt={}
for nm,cfg in (('rules',EventsConfig()),('seq',replace(EventsConfig(),hit_model='seq'))):
    t0=time.time(); res[nm]={c.cid:preds(c,cfg) for c in clips}; tt[nm]=time.time()-t0
    s=LH.score(clips,res[nm]); print(nm,'F1 %.4f P %.3f R %.3f'%(s['f1'],s['precision'],s['recall']),'time %.1fs'%tt[nm])
dur=sum(c.duration_ms for c in clips)/60000; print('video minutes',round(dur,1),'extra seconds per video minute',round((tt['seq']-tt['rules'])/dur,2))
from evaluation import metrics as MM
def cc(c,hs):
    st=MM.eval_hits(c.label,{'hits':hs}); return {'tp':st.tp,'fp':st.fp,'fn':st.fn}
cnt={nm:{c.cid:cc(c,res[nm][c.cid]) for c in clips} for nm in res}
match={c.cid:idx[c.cid]['match_id'] for c in clips}; M=sorted(set(match.values()))
def F(nm,ms):
    ks=[c for c in cnt[nm] if match[c] in ms]; tp=sum(cnt[nm][k]['tp'] for k in ks); fp=sum(cnt[nm][k]['fp'] for k in ks); fn=sum(cnt[nm][k]['fn'] for k in ks)
    return 2*tp/max(2*tp+fp+fn,1)
rng=random.Random(0); ds=sorted(F('seq',s)-F('rules',s) for s in ([rng.choice(M) for _ in M] for _ in range(5000)))
print('matches',len(M),'diff %+.4f per-match 95%% CI [%+.4f, %+.4f]'%(F('seq',M)-F('rules',M),ds[125],ds[4875]))
for mm in M: print(' match',mm,'rules %.3f seq %.3f'%(F('rules',[mm]),F('seq',[mm])))
