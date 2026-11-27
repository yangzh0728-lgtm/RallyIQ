"""d_index.py TAG: per-sheet counts for the fresh-video review sheets (review/d) from my labels and the pipeline preds."""
import json,os,re
B6=os.path.expanduser('~/Desktop/RallyIQ/train_v6'); import sys; tag=sys.argv[1] if len(sys.argv)>1 else 'v6'
out=['| sheet | rows | hits | pipeline hits | missed hits | false hits | unsure | next-court shuttle tracked | note |','|---|---|---|---|---|---|---|---|---|']
T=[0]*7
for k in ('own_20260529_0005','own_20260529_0007','own_20260529_0009'):
    L={int(l.split('|')[0]):[x.strip() for x in l.split('|')] for l in open(f'{B6}/labels/d/{k}.txt') if l[0].isdigit()}
    items=json.load(open(f'{B6}/review/d/{k}_items.json'))
    P=json.load(open(f'{B6}/ckpt/d/{tag}/{k}.json'))['preds']
    pos=[float(v[2]) for v in L.values() if v[1]=='hit']
    for s in range(0,len(items),5):
        rows=items[s:s+5]; c=[0]*7; notes=[]
        for it in rows:
            n,lab,t=it['n'],L[it['n']][1],float(L[it['n']][2]); why=L[it['n']][4]
            pred=any(abs(p['t']-it['t'])<=.2 or abs(p['t']-t)<=.2 for p in P)
            c[0]+=1; c[1]+=lab=='hit'; c[2]+=pred; c[3]+=lab=='hit' and not pred; c[4]+=lab=='not_a_hit' and pred; c[5]+=lab=='unsure_hit'
            nc='next court' in why or "next-court" in why; c[6]+=nc
            if lab=='not_a_hit' and pred: notes.append(f"#{n} false hit: {why[:60]}")
            if lab=='hit' and not pred: notes.append(f"#{n} missed")
        T=[a+b for a,b in zip(T,c)]
        out.append(f"| {k}_{s//5:02d} | "+' | '.join(map(str,c))+f" | {'; '.join(notes)[:160]} |")
out.append('| **total** | '+' | '.join(f'**{x}**' for x in T)+' | |')
print('\n'.join(out))
