import sys; sys.path.insert(0,'/Users/mac/Desktop/RallyIQ/train_v5/scripts')
import pickle,rt,numpy as np,math
from lab import F,O
al=pickle.load(open(rt.B+'/inputs/feats2.pkl','rb')); C={}
def ctx(k):
    if k not in C:
        C[k]=c=rt.Ctx(k);c.byf={f.frame:[p for p in f.players if p.player_id] for f in c.pl.frames}; c.rc=rt.S.RacketContact(c.pl,c.meta.fps,1080)
        # all ids
    return C[k]
def swing(c,f):
    best={}
    fps=c.meta.fps;w=round(.2*fps)
    return None
def feats(c,t):
    fps=c.meta.fps;w=round(.4*fps);n=len(t['xy']);e={m:[] for m in(0,.2,.4)}; side=0
    for f,(x,y) in t['xy']:
        lo={m:1e9 for m in e};hi={m:-1e9 for m in e};sd=set()
        for g in range(f-w,f+w+1,max(1,w//6)):
            for p in c.byf.get(g,()) or c.byf.get(g+1,()):
                h=p.bbox[3]-p.bbox[1]; sd.add(p.side)
                for m in e: lo[m]=min(lo[m],p.bbox[0]-m*h);hi[m]=max(hi[m],p.bbox[2]+m*h)
        side+= len(sd)>=2
        for m in e: e[m].append(max(lo[m]-x,x-hi[m],0)/c.meta.width if hi[m]>=lo[m] else 0)
    # swinging contact at ends/turns
    def con(f,x,y):
        for d in range(-3,4):
            for p in c.byf.get(f+d,()):
                h=max(p.bbox[3]-p.bbox[1],60); ws=[(k[0],k[1]) for i,k in enumerate(p.keypoints) if i in(9,10) and k[2]>.2]
                if ws and min(math.dist(q,(x,y)) for q in ws)<=.65*h:
                    if max(wsp(c,p.player_id,f+d),0)/h>=1.5: return 1
        return 0
    tb=sum(1-con(f,*dict(t['xy'])[f]) for f,_ in t['turns']); tg=len(t['turns'])-tb
    cs=con(t['f0'],*t['xy'][0][1]); ce=con(t['f1'],*t['xy'][-1][1])
    return dict(o0=np.mean([v>0 for v in e[0]]),o2=np.mean([v>0 for v in e[.2]]),o4=np.mean([v>0 for v in e[.4]]),m0=np.mean(e[0]),m4=max(e[.4]),tb=tb,tg=tg,cs=cs,ce=ce,side=side/n,n=n/fps,ds=min(t['ds'],3),de=min(t['de'],3),dmin=min(t['dmin'],3),reg=t['reg'])
def wsp(c,pid,f):
    if not hasattr(c,'wt'):
        c.wt={}
        for g,ps in c.byf.items():
            for p in ps: c.wt.setdefault(p.player_id,{})[g]={i:(p.keypoints[i][0],p.keypoints[i][1]) for i in(9,10) if p.keypoints[i][2]>.2}
        c.wc={}; c.st=2 if len(c.byf)<.6*c.meta.frame_count else 1
    if (pid,f) not in c.wc:
        T=c.wt.get(pid,{});w=round(.15*c.meta.fps);b=0
        for g in range(f-w,f+w+1):
            for s in (1,2):
                A,B=T.get(g-s),T.get(g)
                if A and B:
                    for i in A.keys()&B.keys(): b=max(b,math.dist(A[i],B[i])/s)
                    break
        c.wc[(pid,f)]=b*c.meta.fps
    return c.wc[(pid,f)]
if __name__=='__main__':
    X=[];Y=[];G=[];I=[]
    for i,t in enumerate(al):
        y=1 if i in F else 0
        if t['n']<4: continue
        X.append(feats(ctx(t['key']),t));Y.append(y);G.append(t['key']);I.append((i,i in F or i in O))
    pickle.dump((X,Y,G,I),open(rt.B+'/inputs/clf.pkl','wb'))
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import LeaveOneGroupOut,cross_val_predict
    from sklearn.metrics import roc_auc_score
    names=list(X[0]); M=np.array([[x[k] for k in names] for x in X]);Y=np.array(Y);G=np.array(G)
    lg=make_pipeline(StandardScaler(),LogisticRegression(C=.3,class_weight='balanced'))
    for S in (['o0','o2','o4','m0','m4'],['o0','o4','m0','m4','tb','tg','cs','ce'],names):
        j=[names.index(s) for s in S]; p=cross_val_predict(lg,M[:,j],Y,groups=G,cv=LeaveOneGroupOut(),method='predict_proba')[:,1]
        print(S,'auc',round(roc_auc_score(Y,p),3),[(th,round((p[Y==1]>th).mean(),2),round((p[Y==0]>th).mean(),3)) for th in(.3,.5,.7,.8)])
        lg.fit(M[:,j],Y); print(dict(zip(S,lg[-1].coef_[0].round(2))))
    print([ (I[q][0],G[q][-4:],round(p[q],2)) for q in np.argsort(-p) if Y[q]==0][:40])
    print([ (I[q][0],G[q][-4:],round(p[q],2),{k:round(float(v),2) for k,v in X[q].items()}) for q in np.argsort(p) if Y[q]==1][:20])
