import sys; sys.path.insert(0,'/Users/mac/Desktop/RallyIQ/train_v5/scripts')
import rt,metric,pickle,json
from lab import F,O
from multiprocessing import Pool
al=pickle.load(open(rt.B+'/inputs/feats.pkl','rb'))
def f(a):
    k,ov=a; c=rt.Ctx(k); tr,_=c.run(**ov); pts={p.frame:(p.x,p.y) for p in tr.points}
    s,w=metric.score(k,pts)
    def kept(S):
        n=h=0
        for i in S:
            t=al[i]
            if t['key']!=k: continue
            for g,q in t['xy']:
                n+=1; r=pts.get(g); h+=(r is not None and abs(r[0]-q[0])+abs(r[1]-q[1])<6)
        return h,n
    return k,s,w,kept(F),kept(O),kept([i for i in range(len(al)) if i not in F and i not in O]),len(tr.points)
if __name__=='__main__':
    ov=dict(a.split('=') for a in sys.argv[1:]); ov={k:(v!='0' if k=='own' else float(v)) for k,v in ov.items()}
    T={'ok':0,'wrong':0,'miss':0,'n':0,'badfl':0};K=[[0,0],[0,0],[0,0]]
    with Pool(6) as p:
        for k,s,w,*kk,n in p.imap(f,[(k,ov) for k in rt.IDS]):
            print(k,s,w,kk,n,flush=True)
            for q in T:T[q]+=s[q]
            for a,b in zip(K,kk): a[0]+=b[0];a[1]+=b[1]
    print('TOTAL',T,'foreign frames kept',K[0],'ours(hand) kept',K[1],'rest kept',K[2])
