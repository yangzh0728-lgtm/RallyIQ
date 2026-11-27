"""fill_cands.py BASE FILL DST: base candidates; frames where base has none take FILL's (ft1 fills the base detector's gaps)."""
import sys,json,os
b,f,d=sys.argv[1:4]; os.makedirs(d,exist_ok=True)
for n in os.listdir(b):
    if not n.endswith('.json') or not os.path.exists(f'{f}/{n}'): continue
    A=json.load(open(f'{b}/{n}')); B=json.load(open(f'{f}/{n}'))['frames']; k=0
    for fr,v in B.items():
        if fr not in A['frames'] or not A['frames'][fr]['candidates']: A['frames'][fr]=v; k+=1
    json.dump(A,open(f'{d}/{n}','w')); print(n,k,'frames filled')
