"""thr_cands.py SRC DST THR: keep candidates with score >= THR (the production detector drops peaks below the checkpoint's
recommended threshold, 0.3 for heatmap.pt)."""
import sys,json,os
s,d,t=sys.argv[1],sys.argv[2],float(sys.argv[3]); os.makedirs(d,exist_ok=True)
for n in os.listdir(s):
    if not n.endswith('.json'): continue
    doc=json.load(open(f'{s}/{n}'))
    doc['frames']={f:{**v,'candidates':[c for c in v['candidates'] if c[2]>=t]} for f,v in doc['frames'].items()}
    doc['frames']={f:v for f,v in doc['frames'].items() if v['candidates']}
    json.dump(doc,open(f'{d}/{n}','w'))
