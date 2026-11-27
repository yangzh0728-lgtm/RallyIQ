"""d_score.py TAG KEY...: score ckpt/d/<TAG>/<KEY>.json predictions against labels/d/<KEY>.txt (tol 0.2 s, unsure ignored)."""
import sys,json,os
B6=os.path.expanduser('~/Desktop/RallyIQ/train_v6'); tag=sys.argv[1]; T=[0,0,0]
for k in sys.argv[2:]:
    L=[l.split('|') for l in open(f'{B6}/labels/d/{k}.txt') if l[0].isdigit()]
    pos=[float(x[2]) for x in L if x[1].strip()=='hit']; ign=[float(x[2]) for x in L if x[1].strip()=='unsure_hit']
    P=json.load(open(f'{B6}/../train_v7/ckpt/d/{tag}/{k}.json'))['preds']; pr=[p['t'] for p in P if not any(abs(p['t']-g)<=.2 for g in ign)]
    used=set(); tp=0
    for t in sorted(pr):
        m=[i for i,g in enumerate(pos) if i not in used and abs(t-g)<=.2]
        if m: used.add(min(m,key=lambda i:abs(pos[i]-t))); tp+=1
    print(k,'pos',len(pos),'pred',len(pr),'tp',tp,'F1 %.3f'%(2*tp/max(len(pos)+len(pr),1))); T=[T[0]+len(pos),T[1]+len(pr),T[2]+tp]
print('ALL pos %d pred %d tp %d F1 %.3f'%(T[0],T[1],T[2],2*T[2]/max(T[0]+T[1],1)))
