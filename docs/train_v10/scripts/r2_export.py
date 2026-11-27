"""r2_export.py: choose threshold / gap on ShuttleSet TRAIN out-of-fold probs (LH.score, +-3 frames same side), then export the
3 ShuttleSet-only TCN members + normalisation + decode choice -> repo/rallyiq/events/models/seq_hits.npz"""
import os,sys,glob,json
import numpy as np, torch
R=os.path.expanduser('~/Desktop/RallyIQ/train_v10/repo'); sys.path[:0]=[R,R+'/scripts']
from pathlib import Path
import learn_hits as LH
from rallyiq.events.seqhits import peaks
B=os.path.expanduser('~/Desktop/RallyIQ/train_v10'); P=np.load(f'{B}/ckpt/r_bcast/probs.npz')
clips=LH.load_clips(Path('/Users/mac/Desktop/RallyIQ/train_v5/inputs/pub_v5/shuttleset_train_v3'),'index.json')
D={c.cid:np.load(f'{B}/ckpt/seq/sstrain__{c.cid}.npz') for c in clips}
best=None
for thr in (0.2,0.3,0.4,0.5,0.6,0.7):
    for gap in (0.2,0.28,0.36):
        hb={}
        for c in clips:
            d=D[c.cid]; h=P[f'sstrain__{c.cid}__hit']; far=P[f'sstrain__{c.cid}__far']; fps=float(d['fps'])
            hb[c.cid]=[{'t_ms':c.offset+d['grid'][i]*1000/fps,'side':'far' if far[i]>0.5 else 'near','ok':True} for i in peaks(d['grid'],h,fps,thr,gap)]
        f=LH.score(clips,hb)['f1']; print(thr,gap,round(f,4))
        if best is None or f>best[0]: best=(f,thr,gap)
print('chosen on ShuttleSet train OOF',best)
nm=np.load(f'{B}/ckpt/r_bcast/norm.npz'); out={'mu':nm['mu'],'sd':nm['sd'],'thr':np.array(best[1]),'gap_s':np.array(best[2]),'n_members':np.array(3)}
for s in range(3):
    st=torch.load(f'{B}/ckpt/r_bcast/seed{s}.pt',map_location='cpu')
    for k,v in st.items(): out[f'm{s}/{k}']=v.numpy().astype(np.float32)
np.savez_compressed(f'{R}/rallyiq/events/models/seq_hits.npz',**out); print('exported',os.path.getsize(f'{R}/rallyiq/events/models/seq_hits.npz'))
