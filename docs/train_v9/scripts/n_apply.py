"""n_apply.py: amateur_v6(_newids).json + labels/n_decisions.txt -> labels/amateur_v9(_newids).json.
A decision with a stroke sets stroke (+stroke_sure); '-' leaves the stroke as it was (stroke_sure stays false)."""
import json,os
items=json.load(open('ckpt/n_items.json')); D={}
for l in open('labels/n_decisions.txt'):
    if l[0].isdigit():
        n,s,sure,why=[x.strip() for x in l.split('|',3)]; D[int(n)]=(s,sure=='true',why)
assert len(D)==len(items),(len(D),len(items))
import collections; C=collections.Counter()
for src,dst in (('labels/amateur_v6.json','labels/amateur_v9.json'),('labels/amateur_v6_newids.json','labels/amateur_v9_newids.json')):
    L=json.load(open(src))
    for i,it in enumerate(items):
        h=min(L['hits'][it['k']],key=lambda h:abs(h['t']-it['t'])); assert abs(h['t']-it['t'])<1e-6
        s,sure,why=D[i]
        if s!='-':
            if dst.endswith('v9.json'): C['changed' if h.get('stroke') and h['stroke']!=s else ('new' if not h.get('stroke') else 'confirmed')]+=1
            h['stroke']=s; h['stroke_sure']=sure; h['stroke_review_v9']=why
        elif dst.endswith('v9.json'): C['left']+=1
    L['note']+=' | v9: stroke review of every non-sure am7 hit (labels/n_decisions.txt)'
    json.dump(L,open(dst,'w'),indent=0)
print(C)
for f in ('labels/amateur_v9.json',):
    L=json.load(open(f)); hs=[h for k,v in L['hits'].items() for h in v if h['status'] in('reviewed_sure','reviewed_unsure')]
    print('typed',sum(bool(h.get('stroke')) for h in hs),'sure',sum(bool(h.get('stroke')) and bool(h.get('stroke_sure')) for h in hs),'of',len(hs))
