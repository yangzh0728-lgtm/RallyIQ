"""apply_v6.py: amateur_v4.json + labels/v6_decisions.txt -> labels/amateur_v6.json (am7 only)."""
import json,os
B6=os.path.expanduser('~/Desktop/RallyIQ/train_v6'); L=json.load(open('/Users/mac/Desktop/RallyIQ/train_v5/labels/amateur_v4.json'))
st={'hit':'reviewed_unsure','not_a_hit':'not_a_hit','unsure_hit':'unsure_hit'}; n=0
for l in open(f'{B6}/labels/v6_decisions.txt'):
    if l.startswith('#') or not l.strip(): continue
    k,t,lab,side,why,src=[x.strip() for x in l.split('|')]
    t=float(t); hs=L['hits'][k]
    assert not any(abs(h['t']-t)<=.2 for h in hs), (k,t)
    hs.append({'t':t,'status':st[lab],'side':side if side in('near','far') else None,'hitter':None,'stroke':None,'note':why,'review':'claude_v6','source':src}); n+=1
    hs.sort(key=lambda h:h['t'])
L['note']+=' | v6: unlabelled detections of re-detected tracks reviewed (labels/v6_decisions.txt)'
json.dump(L,open(f'{B6}/labels/amateur_v6.json','w'),indent=0); print(n,'added')
