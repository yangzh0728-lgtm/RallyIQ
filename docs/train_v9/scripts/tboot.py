"""tboot.py T.json A B [T2.json]: paired per-video bootstrap of F1(B)-F1(A) from tune outputs (A,B config names)."""
import sys,json,random
T=json.load(open(sys.argv[1])); 
if len(sys.argv)>4: 
    for k,v in json.load(open(sys.argv[4])).items(): T.setdefault(k,{}).update(v)
A,B=sys.argv[2],sys.argv[3]; QA={'own_20260529_0003','own_20260529_0004','own_20260529_0006','own_20260529_0008','own_20260906_0615','own_20260906_0620'}
def f1(ks,n): t=[sum(T[k][n][x] for k in ks) for x in('tp','fp','fn')]; return 2*t[0]/max(2*t[0]+t[1]+t[2],1)
rng=random.Random(0)
for name,ks in (('all 17',sorted(T)),('other 11',[k for k in T if k not in QA]),('qa6',[k for k in T if k in QA])):
    d=sorted(f1(s,B)-f1(s,A) for s in ([rng.choice(ks) for _ in ks] for _ in range(5000)))
    print(f'{name}: {f1(ks,A):.4f} -> {f1(ks,B):.4f} diff {f1(ks,B)-f1(ks,A):+.4f} CI [{d[125]:+.4f}, {d[4875]:+.4f}]')
