"""boot5.py LABELS A B: paired bootstrap over videos of hit F1 (B - A) from ckpt/preds."""
import sys,json,random; sys.path.insert(0,'/Users/mac/Desktop/RallyIQ/train_v5/scripts')
import rt
from pathlib import Path
import amateur_hits as AH
lab=Path(rt.B)/sys.argv[1]; clips=AH.load_clips(Path(rt.B)/'inputs_C',lab)
pa,pb=(json.load(open(f'{rt.B}/ckpt/preds/{n}.json')) for n in sys.argv[2:4])
A={c.cid:AH.score_clip(c,pa[c.cid]) for c in clips};Bv={c.cid:AH.score_clip(c,pb[c.cid]) for c in clips}
def f1(r):
    tp=sum(x['tp'] for x in r);fp=sum(x['fp'] for x in r);pos=sum(x['pos'] for x in r); return 2*tp/max(2*tp+fp+pos-tp,1)
rng=random.Random(0)
for name,vids in (('qa6',[c.cid for c in clips if c.truth['set']=='qa6']),('other 11',[c.cid for c in clips if c.truth['set']!='qa6']),('all 17',[c.cid for c in clips])):
    d0=f1([Bv[v] for v in vids])-f1([A[v] for v in vids]);ds=sorted(f1([Bv[v] for v in s])-f1([A[v] for v in s]) for s in ([rng.choice(vids) for _ in vids] for _ in range(5000)))
    print(f"{sys.argv[2]} -> {sys.argv[3]} {name}: {f1([A[v] for v in vids]):.3f} -> {f1([Bv[v] for v in vids]):.3f} diff {d0:+.3f} 95% CI [{ds[125]:+.3f}, {ds[4875]:+.3f}]")
