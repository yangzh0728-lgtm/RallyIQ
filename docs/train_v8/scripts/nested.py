"""nested.py T.json BASE [prefix]: nested selection of a config: for each am7 video, the config with the best F1 on the
other 10 am7 videos is applied to it; qa6 gets the config best on all 11 am7. Paired bootstrap vs BASE."""
import sys,json,random
T=json.load(open(sys.argv[1])); base=sys.argv[2]; pre=sys.argv[3] if len(sys.argv)>3 else ''
QA={'own_20260529_0003','own_20260529_0004','own_20260529_0006','own_20260529_0008','own_20260906_0615','own_20260906_0620'}
AM=[k for k in T if k not in QA]; names=[n for n in T[AM[0]] if n.startswith(pre) or n==base]
def F(c): return 2*c[0]/max(2*c[0]+c[1]+c[2],1)
def tot(ks,n): return [sum(T[k][n][x] for k in ks) for x in('tp','fp','fn')]
def best(ks): return max(names,key=lambda n:(F(tot(ks,n)),n==base))
sel={v:best([k for k in AM if k!=v]) for v in AM}; full=best(AM)
for k in QA: sel[k]=full
print('chosen per held-out video',sorted(set(sel.values())),'full',full)
N={k:T[k][sel[k]] for k in T}; Bv={k:T[k][base] for k in T}
f=lambda D,ks:F([sum(D[k][x] for k in ks) for x in('tp','fp','fn')])
rng=random.Random(0)
for nm,ks in (('all 17',list(T)),('am7 11',AM),('qa6',[k for k in T if k in QA])):
    d=sorted(f(N,s)-f(Bv,s) for s in ([rng.choice(ks) for _ in ks] for _ in range(5000)))
    print(f'{nm}: {f(Bv,ks):.4f} -> {f(N,ks):.4f} diff {f(N,ks)-f(Bv,ks):+.4f} CI [{d[125]:+.4f}, {d[4875]:+.4f}]')
