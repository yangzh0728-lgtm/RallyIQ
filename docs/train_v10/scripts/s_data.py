"""s_data.py: stroke windows -> ckpt/s_data.npz. Each row: 46 grid frames (-15..+30) x 27 seq features + side flag channel.
groups: ss_train / ss_test (ShuttleSet csv strokes matched to labelled hits within 60 ms), own (v9 stroke rows: key,t,y,sure),
fresh (v8 fresh items + my stroke labels)."""
import os,sys,json,csv,pickle,glob
import numpy as np
R=os.path.expanduser('~/Desktop/RallyIQ/train_v10/repo'); sys.path[:0]=[R]
from rallyiq.strokes.taxonomy import from_shuttleset
from rallyiq.strokes import court as CM
CL=CM.get_model().classes
B=os.path.expanduser('~/Desktop/RallyIQ/train_v10'); SEQ=f'{B}/ckpt/seq'
def window(name,t,side):
    d=np.load(f'{SEQ}/{name}.npz'); st=d['grid'][1]-d['grid'][0]; i=int(round(t*float(d['fps'])/st)); X=d['X']
    W=np.zeros((46,28),np.float32)
    for j in range(46):
        q=i-15+j
        if 0<=q<len(X): W[j,:27]=X[q]
    W[:,27]=1.0 if side=='far' else 0.0
    return W
rows=[]
for split,root in (('ss_train','shuttleset_train_v3'),('ss_test','shuttleset_test_v3')):
    idx=json.load(open(f'/Users/mac/Desktop/RallyIQ/train_v5/inputs/pub_v5/{root}/index.json'))
    for cid,m in idx.items():
        if "video" not in m: continue
        p=f'/Users/mac/Desktop/RallyIQ/train_v4/inputs/shuttleset_set/{m["video"]}/{cid.split("_")[1]}.csv'
        name=f"{'sstrain' if split=='ss_train' else 'sstest'}__{cid}"
        if not os.path.exists(p) or not os.path.exists(f'{SEQ}/{name}.npz'): continue
        st=[(float(r['frame_num'])*1000/m['fps'],from_shuttleset((r.get('type') or '').strip())[0]) for r in csv.DictReader(open(p,encoding='utf-8')) if r.get('frame_num')]
        lab=json.loads(open(f'/Users/mac/Desktop/RallyIQ/train_v5/inputs/pub_v5/{root}/labels/{cid}.json',encoding='latin-1').read())
        off=lab['range_ms'][0]
        for h in lab['hits']:
            c=min(st,key=lambda x:abs(x[0]-h['t_ms']))
            if abs(c[0]-h['t_ms'])<=60 and c[1] in CL:
                rows.append({'g':split,'key':cid,'t':(h['t_ms']-off)/1000,'y':c[1],'sure':True,'W':window(name,(h['t_ms']-off)/1000,h.get('side'))})
own=pickle.load(open('/Users/mac/Desktop/RallyIQ/train_v9/ckpt/n_rows.pkl','rb'))['rows']
for r in own:
    if r['y'] in CL: rows.append({'g':'own','key':r['key'],'t':r['t'],'y':r['y'],'sure':r['sure'],'W':window('own__'+r['key'],r['t'],r.get('side'))})
items=json.load(open('/Users/mac/Desktop/RallyIQ/train_v8/ckpt/stroke_items.json'))
lab={}
for l in open('/Users/mac/Desktop/RallyIQ/train_v8/labels/fresh_strokes.txt'):
    if l[0].isdigit(): n,s,sure,_=[x.strip() for x in l.split('|',3)]; lab[int(n)]=(s,sure=='true')
for i,it in enumerate(items):
    if lab[i][0] in CL: rows.append({'g':'fresh','key':it['k'],'t':it['t'],'y':lab[i][0],'sure':lab[i][1],'n':i,'W':window('fresh__'+it['k'],it['t'],it['side'])})
np.savez_compressed(f'{B}/ckpt/s_data.npz',W=np.stack([r['W'] for r in rows]),y=np.array([CL.index(r['y']) for r in rows]),g=np.array([r['g'] for r in rows]),
                    key=np.array([r['key'] for r in rows]),t=np.array([r['t'] for r in rows]),sure=np.array([r['sure'] for r in rows]),n=np.array([r.get('n',-1) for r in rows]))
import collections; print(collections.Counter((r['g'],r['sure']) for r in rows)); print(CL)
