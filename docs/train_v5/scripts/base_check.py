import sys; sys.path.insert(0,'/Users/mac/Desktop/RallyIQ/train_v5/scripts')
import rt,time
from multiprocessing import Pool
def f(k):
    t=time.time();c=rt.Ctx(k); tr,_=c.run(); a={p.frame:(p.x,p.y) for p in c.old.points}; b={p.frame:(p.x,p.y) for p in tr.points}
    same=sum(1 for q in a if q in b and abs(a[q][0]-b[q][0])+abs(a[q][1]-b[q][1])<2)
    return k,len(a),len(b),same,len(c.cands),round(time.time()-t,1),min(s for v in c.cands.values() for *_,s in v)
if __name__=='__main__':
    with Pool(6) as p:
        for r in p.imap(f,rt.IDS): print(*r,flush=True)
