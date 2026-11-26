import sys,json; sys.path.insert(0,'/Users/mac/Desktop/RallyIQ/train_v5/scripts')
import rt
from pathlib import Path
import amateur_hits as AH
A,B=sys.argv[1:3]; keys=sys.argv[3:]
P4=Path(rt.H+'/train_v4'); TR=AH.load_truth(P4/'inputs',P4/'labels/amateur_v3.json')
pa,pb=(json.load(open(f'{rt.B}/ckpt/preds/{n}.json')) for n in (A,B))
def ts(p): return sorted((x.get('t') if 't' in x else x['t_ms']/1000, x.get('side'),x.get('hitter')) for x in p)
def match(pr,k):
    pos=[l for l in TR[k]['labels'] if l['kind']=='pos']; used=set(); hit={}
    for d,i,j in sorted((abs(p[0]-l['t']),i,j) for i,p in enumerate(pr) for j,l in enumerate(pos)):
        if d<=.25 and i not in used and j not in hit: used.add(i);hit[j]=i
    ign=[l['t'] for l in TR[k]['labels'] if l['kind']=='ign']; neg=[l['t'] for l in TR[k]['labels'] if l['kind']=='neg']
    fp=[(round(p[0],2),p[1],'neg' if any(abs(p[0]-x)<.25 for x in neg) else '') for i,p in enumerate(pr) if i not in used and any(a<=p[0]<=b for a,b in TR[k]['windows']) and not any(abs(p[0]-x)<.25 for x in ign)]
    return {round(pos[j]['t'],2) for j in hit},fp
for k in keys or TR:
    def get(p):
        p=p[k]; 
        if isinstance(p,dict): p=p.get('shots') or p.get('hits')
        return ts(p)
    ha,fa=match(get(pa),k);hb,fb=match(get(pb),k)
    if ha!=hb or fa!=fb: print(k,'LOST',sorted(ha-hb),'GAINED',sorted(hb-ha),'\n   newFP',[x for x in fb if not any(abs(x[0]-y[0])<.12 for y in fa)],'\n   goneFP',[x for x in fa if not any(abs(x[0]-y[0])<.12 for y in fb)])
