import sys,os; sys.path.insert(0,'/Users/mac/Desktop/RallyIQ/train_v5/scripts')
import sheets5 as S,rt,json
from pathlib import Path
import amateur_hits as AH
"""nf.py PREDS key... : sheets for predictions that match no label (N) and missed positives that old_v3b found (L)"""
P=sys.argv[1];P4=Path(rt.H+'/train_v4'); TR=AH.load_truth(P4/'inputs',Path(rt.B+'/labels/amateur_v4.json'))
pa,pb=(json.load(open(f'{rt.B}/ckpt/preds/{n}.json')) for n in (os.environ.get('BASE','old4f'),P)); OUT=os.environ.get('OUT','nf')
DONE={}
for l in open(rt.B+'/labels/v5_decisions.txt'):
    if l[0]!='#': q=[x.strip() for x in l.split('|')]; DONE.setdefault(q[0],[]).append(float(q[2]) if q[1]!='retime' else float(q[3]))
for k in sys.argv[2:]:
    pr=sorted(pb[k],key=lambda p:p['t']); pos=[l for l in TR[k]['labels'] if l['kind']=='pos'];lab=TR[k]['labels']
    its=[]
    inw=lambda t:any(a<=t<=b for a,b in TR[k]['windows'])
    for i,p in enumerate(pr):
        if inw(p['t']) and not any(abs(p['t']-l['t'])<=.25 and l['kind']!='neg' for l in lab) and not any(abs(p['t']-d)<.2 for d in DONE.get(k,())):
            its.append((p['t'],p['side'],f"NEW {p['hitter']}/{p['side']} conf {p['conf']:.2f}",p['hitter']))
    for l in pos:
        if not any(abs(p['t']-l['t'])<=.25 for p in pr) and any(abs(p['t']-l['t'])<=.25 for p in pa[k]): its.append((l['t'],l['side'],'LOST label',None))
    its.sort(); ts=sorted([p['t'] for p in pr]); rows=[]
    for t,s,tt,h in its:
        n=[x for x in ts if x>t+.2]; rows.append(dict(key=k,t=t,side=s,hitter=h,next_t=n[0] if n and n[0]-t<2.5 else None,title=f"{k[-7:]} t={t:.2f} {tt} | labels near: {[ (round(l['t'],2),l['kind'],l['side']) for l in lab if abs(l['t']-t)<1.2]}"))
    print(k,len(rows),S.render(rows,f'{rt.B}/review/{OUT}',k,per=4))
