"""hacc.py PREDS [LABELS]: hitter accuracy on labelled doubles hits (pred matched within 0.2 s)."""
import sys,json
P=json.load(open(f'/Users/mac/Desktop/RallyIQ/train_v5/ckpt/preds/{sys.argv[1]}.json'))
L=json.load(open(sys.argv[2] if len(sys.argv)>2 else '/Users/mac/Desktop/RallyIQ/train_v5/labels/amateur_v4_newids.json'))
D=['own_20260310_01','own_20260320_0239','own_20260324_0241','own_20260906_0622']
T=[0]*5
for k in D:
    hs=[h for h in L['hits'][k] if h['status'] in('reviewed_sure','reviewed_unsure')]; c=[0]*5
    for h in hs:
        m=min(P[k],key=lambda p:abs(p['t']-h['t']));
        if abs(m['t']-h['t'])>0.2: c[0]+=1; continue
        if h.get('hitter') is None: c[1]+=1; continue
        c[2]+=1; c[3]+=m['hitter']==h['hitter']; c[4]+=m['side']==h['side']
    print(k,len(hs),'missed',c[0],'untracked-label',c[1],'hitter',f'{c[3]}/{c[2]}','side',c[4]); T=[a+b for a,b in zip(T,c)]
print('ALL missed',T[0],'untracked',T[1],'hitter',f'{T[3]}/{T[2]}',round(T[3]/max(T[2],1),3),'side',T[4])
