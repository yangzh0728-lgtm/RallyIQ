import sys; sys.path.insert(0,'/Users/mac/Desktop/RallyIQ/train_v5/scripts')
import pickle,rt,mosaic,sheets,cv2,numpy as np
from lab import F,O
al=pickle.load(open(rt.B+'/inputs/feats.pkl','rb'));D=sys.argv[1];out=[];V={}
N={}
tot=[]
for i,t in enumerate(al):
    if i in F or t['n']<6: continue
    k=t['key']
    if k not in N: N[k]={p.frame:(p.x,p.y) for p in rt.ShuttleTrack.model_validate_json(rt.rd(f'{rt.B}/{D}/new_runs/{k}','shuttle')).points}
    lost=[g for g,q in t['xy'] if not(g in N[k] and abs(N[k][g][0]-q[0])+abs(N[k][g][1]-q[1])<8)]
    if len(lost)>=max(5,.25*t['n']): tot.append((i,t,len(lost)))
print(len(tot),sum(x[2] for x in tot))
ims=[]
for i,t,l in tot:
    k=t['key']
    if k not in V: V[k]=(sheets.Video(k),rt.Ctx(k))
    im=mosaic.tile(V[k][0],t,V[k][1],f'{i}{"O" if i in O else ""} {k[-7:]} lost{l}'); ims.append(im)
for k in range(0,len(ims),24):
    b=ims[k:k+24]
    while len(b)%4: b.append(np.zeros((270,480,3),np.uint8))
    cv2.imwrite(f'{rt.B}/review/tk/lost_{k//24}.jpg',cv2.vconcat([cv2.hconcat(b[j:j+4]) for j in range(0,len(b),4)]),[1,82])
