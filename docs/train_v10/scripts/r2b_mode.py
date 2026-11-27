import os,sys,json
import numpy as np
R=os.path.expanduser('~/Desktop/RallyIQ/train_v10/repo'); sys.path[:0]=[R,R+'/scripts']
from pathlib import Path
import learn_hits as LH
from rallyiq.events.config import EventsConfig
from rallyiq.events.seqhits import peaks
B=os.path.expanduser('~/Desktop/RallyIQ/train_v10'); P=np.load(f'{B}/ckpt/r_bcast/probs.npz'); m=np.load(f'{R}/rallyiq/events/models/seq_hits.npz')
thr,gap=float(m['thr']),float(m['gap_s'])
clips=LH.load_clips(Path('/Users/mac/Desktop/RallyIQ/train_v5/inputs/pub_v5/shuttleset_train_v3'),'index.json')
res={'r03':{},'r10':{},'all':{}}
for c in clips:
    d=np.load(f'{B}/ckpt/seq/sstrain__{c.cid}.npz'); fps=float(d['fps']); h=P[f'sstrain__{c.cid}__hit']; far=P[f'sstrain__{c.cid}__far']
    spans=[(r.start_ms,r.end_ms) for r in c.run(EventsConfig())]
    pk=[(d['grid'][i]*1000/fps,'far' if far[i]>.5 else 'near') for i in peaks(d['grid'],h,fps,thr,gap)]
    for nm,mg in (('r03',300),('r10',1000),('all',None)):
        res[nm][c.cid]=[{'t_ms':c.offset+t,'side':s,'ok':True} for t,s in pk if mg is None or any(a-mg<=t<=b+mg for a,b in spans)]
for nm in res: print(nm,round(LH.score(clips,res[nm])['f1'],4))
