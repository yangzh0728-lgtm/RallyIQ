"""(a) right shuttle at the hit but the drawn trail includes a next-court trajectory; (b) the shuttle at the hit (or at the
NEXT panel) is a next-court one. Next-court = a hand-labelled foreign tracklet (lab.F). old: v4 tracks + old trail drawing,
new: inputs_C tracks + trail of the hit's own flight."""
import sys,os,json,pickle; sys.path.insert(0,'/Users/mac/Desktop/RallyIQ/train_v5/scripts')
import rt,numpy as np,sheet_items,importlib
from lab import F
al=pickle.load(open(rt.B+'/inputs/feats.pkl','rb'))
FP={}
for i in F:
    for f,(x,y) in al[i]['xy']: FP.setdefault((al[i]['key'],f),[]).append((x,y))
def foreign(k,f,x,y): return any(abs(x-a)+abs(y-b)<8 for a,b in FP.get((k,f),[]))
os.environ['INP']='inputs_C'; import sheets5 as S
def oldtrail(v,t0,t1): return [(f,v.sh[f]['x'],v.sh[f]['y']) for f in range(round(t0*v.fps),round(t1*v.fps)+1) if v.sh.get(f) and v.sh[f].get('x') is not None]
class Old(S.Video):
    def __init__(s,k):
        s.key=k;a=rt.art(k);s.fps=float(rt.IDS[k][2]);s.sh={p['frame']:p for p in json.loads(rt.rd(a,'shuttle'))['points']}
class New(Old):
    def __init__(s,k):
        s.key=k;s.fps=float(rt.IDS[k][2]);s.sh={p['frame']:p for p in json.load(open(f'{rt.B}/inputs_C/new_runs/{k}/shuttle.json'))['points']}
res={};tot={'old':[0,0,0],'new':[0,0,0]}
for k,its in sheet_items.items().items():
    vo,vn=Old(k),New(k)
    for it in its:
        t=it['t'];nt=min(it.get('next_t') or t+1.5,t+2.5);r={}
        for name,v,tr in (('old',vo,lambda v,a,b,c:oldtrail(v,a,b)),('new',vn,S.flight)):
            pts=tr(v,t-.3,nt,t)+tr(v,t,nt+.3,nt)
            g=[(round(q*v.fps),v.sh.get(round(q*v.fps))) for q in (t,nt)]
            b=any(p and p.get('x') is not None and foreign(k,f,p['x'],p['y']) for f,p in g)
            a=(not b) and sum(foreign(k,f,x,y) for f,x,y in pts)>=3
            r[name]=('b' if b else 'a' if a else '')
            tot[name][0]+=a;tot[name][1]+=b;tot[name][2]+=1
        res.setdefault(it['sheet'],[]).append((it['row'],it['id'],r['old'],r['new']))
json.dump(res,open(rt.B+'/review/ab_counts.json','w'),indent=0)
print('old (a,b,rows)',tot['old'],'new',tot['new'])
for sh,v in sorted(res.items()):
    if any(x[2] or x[3] for x in v): print(sh,[x for x in v if x[2] or x[3]])
