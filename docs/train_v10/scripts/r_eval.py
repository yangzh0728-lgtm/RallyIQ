"""r_eval.py TAG: decode the sequence model (peak picking, threshold chosen nested on the other am7 videos' out-of-fold
probabilities), score own videos with amateur_hits.score_clip against the v7 pipeline (ckpt t from train_v9 tP.json 'v7'),
fresh set and ShuttleSet test (LH.score, +-3 frames same side as the gate)."""
import os,sys,json,random,glob
import numpy as np
R=os.path.expanduser('~/Desktop/RallyIQ/train_v10/repo'); sys.path[:0]=[R,R+'/scripts']
from pathlib import Path
import amateur_hits as AH, learn_hits as LH
B=os.path.expanduser('~/Desktop/RallyIQ/train_v10'); tag=sys.argv[1]
P=np.load(f'{B}/ckpt/r_{tag}/probs.npz'); D={os.path.basename(p)[:-4]:np.load(p) for p in glob.glob(f'{B}/ckpt/seq/*.npz')}
QA={'own_20260529_0003','own_20260529_0004','own_20260529_0006','own_20260529_0008','own_20260906_0615','own_20260906_0620'}
INP=Path('/Users/mac/Desktop/RallyIQ/train_v5/inputs_C'); TR=AH.load_truth(INP,Path(f'{B}/labels/amateur_v6.json'))
sys.path.insert(0,B+'/scripts'); import r_feats as RF
for k in ('own_20260529_0005','own_20260529_0007','own_20260529_0009'): TR[k]=RF.fresh_truth(k)
def decode(k,thr,gap=0.28):
    h=P[f'{k}__hit']; far=P[f'{k}__far']; d=D[k]; g=d['grid']; fps=float(d['fps'])
    pk=[i for i in range(1,len(h)-1) if h[i]>=thr and h[i]>=h[i-1] and h[i]>=h[i+1]]
    pk.sort(key=lambda i:-h[i]); keep=[]
    for i in pk:
        if all(abs(g[i]-g[j])/fps>=gap for j in keep): keep.append(i)
    return [(g[i]/fps,'far' if far[i]>0.5 else 'near',float(h[i])) for i in sorted(keep)]
class T:  # minimal clip for score_clip
    def __init__(s,truth): s.truth=truth
    def in_window(s,t,pad=0.0): return any(a-pad<=t<=b+pad for a,b in s.truth['windows'])
def counts(k,thr):
    pr=[{'t':t,'hitter':None,'side':sd} for t,sd,_ in decode('own__'+k if not k.startswith('own__') else k,thr)]
    s=AH.score_clip(T(TR[k]),pr); return s['tp'],s['fp'],s['fn']
def F(c): return 2*c[0]/max(2*c[0]+c[1]+c[2],1)
THR=[0.2,0.3,0.4,0.5,0.6,0.7]
AM=sorted(k for k in TR if TR[k]['set']=='am7')
C={k:{t:counts(k,t) for t in THR} for k in AM}
sel={}; res={}
for v in AM:
    oth=[k for k in AM if k!=v]; sel[v]=max(THR,key=lambda t:F([sum(C[k][t][i] for k in oth) for i in range(3)])); res[v]=C[v][sel[v]]
full=max(THR,key=lambda t:F([sum(C[k][t][i] for k in AM) for i in range(3)]))
for k in TR:
    if k not in AM:
        pre='own__' if TR[k]['set']!='fresh' else 'fresh__'
        pr=[{'t':t,'hitter':None,'side':sd} for t,sd,_ in decode(pre+k,full)]; s=AH.score_clip(T(TR[k]),pr); res[k]=(s['tp'],s['fp'],s['fn'])
print('thresholds',sel,'full',full)
BASE=json.load(open(os.path.expanduser('~/Desktop/RallyIQ/train_v9/ckpt/tP.json')))
bv={k:(BASE[k]['v7']['tp'],BASE[k]['v7']['fp'],BASE[k]['v7']['fn']) for k in BASE}
QAk=[k for k in AM+list(QA) if k in QA]; ALL=AM+QAk
for nm,ks in (('am7 11',AM),('qa6',QAk),('all 17',ALL)):
    print(f"{nm}: v7 {F([sum(bv[k][i] for k in ks) for i in range(3)]):.4f}  seq {F([sum(res[k][i] for k in ks) for i in range(3)]):.4f}")
rng=random.Random(0); d=sorted(F([sum(res[k][i] for k in s) for i in range(3)])-F([sum(bv[k][i] for k in s) for i in range(3)]) for s in ([rng.choice(ALL) for _ in ALL] for _ in range(5000)))
print('all 17 diff CI [%.4f, %.4f]'%(d[125],d[4875]))
FR=['own_20260529_0005','own_20260529_0007','own_20260529_0009']; print('fresh seq',round(F([sum(res[k][i] for k in FR) for i in range(3)]),4))
# ShuttleSet test
clips=LH.load_clips(Path('/Users/mac/Desktop/RallyIQ/train_v5/inputs/pub_v5/shuttleset_test_v3'),'results.json')
hb={c.cid:[{'t_ms':c.offset+t*1000,'side':sd,'ok':True} for t,sd,_ in decode('sstest__'+c.cid,full)] for c in clips}
s=LH.score(clips,hb); print('ShuttleSet test seq F1',round(s['f1'],4),'P',round(s['precision'],3),'R',round(s['recall'],3))
json.dump({'res':res,'sel':sel,'full':full},open(f'{B}/ckpt/r_{tag}/eval.json','w'))
