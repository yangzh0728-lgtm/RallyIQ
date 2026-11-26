import sys; sys.path.insert(0,'/Users/mac/Desktop/RallyIQ/train_v5/scripts'); sys.path.insert(0,'/Users/mac/Desktop/RallyIQ/train_v4/scripts')
import pickle,cv2,numpy as np,sheets,rt,json
from multiprocessing import Pool
def tile(v,t,c,i,tag=''):
    f=(t['f0']+t['f1'])//2; im=v.frame(f/v.fps)
    if im is None: im=np.zeros((1080,1920,3),np.uint8)
    im=im.copy()
    for p in v.players(f):
        if p.get('player_id'):
            b=list(map(int,p['bbox'])); cv2.rectangle(im,b[:2],b[2:],(255,200,0),3)
    for g in range(t['f0'],t['f1']+1):
        for x,y,s in c.cands.get(g,[]): cv2.circle(im,(int(x),int(y)),5,(255,255,0),-1)
    P=[q for _,q in t['xy']]
    for a,b in zip(P,P[1:]): cv2.line(im,tuple(map(int,a)),tuple(map(int,b)),(0,120,255),5)
    cv2.circle(im,tuple(map(int,P[0])),14,(0,255,0),4); cv2.circle(im,tuple(map(int,P[-1])),14,(0,0,255),4)
    im=cv2.resize(im,(480,270))
    cv2.putText(im,f"#{i} {t['f0']/v.fps:.2f}-{t['f1']/v.fps:.2f} n{t['n']} s{t['ds']:.1f} e{t['de']:.1f} T{','.join(str(d) for _,d in t['turns'][:3])}",(3,14),0,.42,(0,0,0),3)
    cv2.putText(im,f"#{i} {t['f0']/v.fps:.2f}-{t['f1']/v.fps:.2f} n{t['n']} s{t['ds']:.1f} e{t['de']:.1f} T{','.join(str(d) for _,d in t['turns'][:3])}",(3,14),0,.42,(0,255,255),1)
    return im
def do(a):
    key,items,out=a
    v=sheets.Video(key); c=rt.Ctx(key); fs=[]
    for k in range(0,len(items),24):
        ts=[tile(v,t,c,i) for i,t in items[k:k+24]]
        while len(ts)%4: ts.append(np.zeros((270,480,3),np.uint8))
        img=cv2.vconcat([cv2.hconcat(ts[j:j+4]) for j in range(0,len(ts),4)])
        fn=f'{out}/{key}_{k//24:02d}.jpg'; cv2.imwrite(fn,img,[1,82]); fs.append(fn)
    return key,len(items),fs
if __name__=='__main__':
    al=pickle.load(open(rt.B+'/inputs/feats.pkl','rb')); by={}
    for i,t in enumerate(al): by.setdefault(t['key'],[]).append((i,t))
    jobs=[]
    for k,its in by.items():
        sus=[(i,t) for i,t in its if (t['ds']>.6 and t['de']>.6) or any(d>.8 for _,d in t['turns']) or t['dmin']>.6]
        jobs.append((k,sus,rt.B+'/review/tk'))
    with Pool(4) as p:
        for r in p.imap(do,jobs): print(*r[:2],len(r[2]),flush=True)
