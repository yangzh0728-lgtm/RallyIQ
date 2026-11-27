"""combine.py MODE DST: base (shipped) candidates combined with ft1's. fill = ft1 only where base has no candidate >= 0.3;
union = both sets, ft1 peaks within 6 px of a base peak dropped."""
import sys,json,os,math
mode,d=sys.argv[1],sys.argv[2]; os.makedirs(d,exist_ok=True)
B='/Users/mac/Desktop/RallyIQ/train_v5/inputs/cands'; F=os.path.expanduser('~/Desktop/RallyIQ/train_v7/ckpt/cands/ft1')
for n in os.listdir(F):
    if not n.endswith('.json') or not os.path.exists(f'{B}/{n}'): continue
    A=json.load(open(f'{B}/{n}')); X=json.load(open(f'{F}/{n}'))['frames']
    for f,v in X.items():
        fc=[c for c in v['candidates'] if c[2]>=.3]
        if not fc: continue
        a=A['frames'].get(f); ac=[c for c in (a['candidates'] if a else []) if c[2]>=.3]
        if mode=='fill':
            if not ac: A['frames'][f]={'candidates':fc,'selected':-1}
        else:
            add=[c for c in fc if all(math.hypot(c[0]-b[0],c[1]-b[1])>6 for b in ac)]
            if add: A['frames'][f]={'candidates':sorted(ac+add,key=lambda c:-c[2])[:5],'selected':-1}
    json.dump(A,open(f'{d}/{n}','w'))
