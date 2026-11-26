import sys,json,math; sys.path.insert(0,'/Users/mac/Desktop/RallyIQ/train_v5/scripts')
import rt
from collections import defaultdict
FL=defaultdict(list)
for x in json.load(open(rt.B+'/inputs/flights.json')): FL[x['key']].append(x)
def score(k,pts):
    """pts {frame:(x,y)} -> ok, wrong(>60px), missing, total, flights mostly wrong(>30% wrong)"""
    ok=wr=mi=n=bad=0; W=[]
    for fl in FL[k]:
        w=0
        for f,(x,y) in fl['pts'].items():
            n+=1; q=pts.get(int(f))
            if q is None: mi+=1; continue
            d=math.dist(q,(x,y))
            if d<=25: ok+=1
            elif d>60: wr+=1; w+=1
        if w>.3*len(fl['pts']): bad+=1; W.append((round(fl['t'],2),w,len(fl['pts'])))
    return dict(ok=ok,wrong=wr,miss=mi,n=n,badfl=bad,fl=len(FL[k])),W
