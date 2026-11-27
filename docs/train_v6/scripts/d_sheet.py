"""d_sheet.py KEY [TAG]: review rows for every pipeline hit (P) and rules candidate (R) inside the fixed windows.
Row = 3 frames (-0.2 / 0 / +0.2 s), track points of the last 0.25 s (yellow), current point (cyan ring), player boxes.
-> review/d/<KEY>_NN.jpg (5 rows each) + review/d/<KEY>_items.json"""
import sys,os,json,cv2,numpy as np
B6=os.path.expanduser('~/Desktop/RallyIQ/train_v6'); k=sys.argv[1]; tag=sys.argv[2] if len(sys.argv)>2 else 'v6'
art=f'{B6}/inputs/d_run/new_runs/{k}'; D=json.load(open(f'{B6}/ckpt/d/{tag}/{k}.json'))
fps=json.load(open(f'{art}/probe.json'))['fps']
S={p['frame']:p for p in json.load(open(f'{art}/shuttle.json'))['points']}
PL={f['frame']:f['players'] for f in json.load(open(f'{art}/players.json'))['frames']}
items=[]
for src,lst in (('P',D['preds']),('R',D['rules'])):
    for p in lst:
        m=next((it for it in items if abs(it['t']-p['t'])<=.2),None)
        if m: m['src']+=src; m.setdefault('side_'+src,p['side'])
        else: items.append({'t':p['t'],'src':src,'side_'+src:p['side']})
items.sort(key=lambda x:x['t'])
for i,it in enumerate(items): it['n']=i
json.dump(items,open(f'{B6}/review/d/{k}_items.json','w'),indent=0)
cap=cv2.VideoCapture(f'{art}/source.mp4'); rows=[]
def panel0(t,title,box=None):
    f=round(t*fps); cap.set(1,f); ok,im=cap.read()
    for d in (0,-1,1,-2,2):
        if f+d in PL:
            for p in PL[f+d]:
                b=list(map(int,p['bbox'])); col=(0,200,0) if p.get('side')=='near' else (255,160,0)
                cv2.rectangle(im,b[:2],b[2:],col,2); cv2.putText(im,(p.get('side') or '?')[0],(b[0],b[1]-6),0,1,col,2)
            break
    for g in range(f-round(.25*fps),f):
        if g in S: cv2.circle(im,(int(S[g]['x']),int(S[g]['y'])),4,(0,255,255),-1)
    if f in S: cv2.circle(im,(int(S[f]['x']),int(S[f]['y'])),14,(255,255,0),2)
    x0,y0,x1,y1=box; im=cv2.resize(im[y0:y1,x0:x1],(640,360)); cv2.putText(im,title,(6,26),0,.8,(255,255,255),2); return im
def cbox(t):
    f=round(t*fps); bs=[]
    for d in range(-6,7):
        bs+= [p['bbox'] for p in PL.get(f+d,[])]
    pts=[(S[g]['x'],S[g]['y']) for g in range(f-int(.3*fps),f+int(.3*fps)) if g in S]
    xs=[b[0] for b in bs]+[b[2] for b in bs]+[p[0] for p in pts]; ys=[b[1] for b in bs]+[b[3] for b in bs]+[p[1] for p in pts]
    if not xs: return (0,0,1920,1080)
    cx,cy=(min(xs)+max(xs))/2,(min(ys)+max(ys))/2; w=max(640,(max(xs)-min(xs))*1.15,(max(ys)-min(ys))*1.15*16/9); h=w*9/16
    x0=int(min(max(cx-w/2,0),1920-w)); y0=int(min(max(cy-h/2,0),1080-h)); return (max(x0,0),max(y0,0),int(max(x0,0)+min(w,1920)),int(max(y0,0)+min(h,1080)))
for it in items:
    t=it['t']; bx=cbox(t); panel=(lambda P: (lambda tt,ti: P(tt,ti,bx)))(panel0); rows.append(np.hstack([panel(t-.2,f"#{it['n']} {t:.2f} {it['src']} {it.get('side_P') or it.get('side_R') or ''} -0.2"),panel(t,'0'),panel(t+.2,'+0.2')]))
for s in range(0,len(rows),5):
    cv2.imwrite(f'{B6}/review/d/{k}_{s//5:02d}.jpg',np.vstack(rows[s:s+5]),[cv2.IMWRITE_JPEG_QUALITY,72])
print(k,len(items),'items',(len(rows)+4)//5,'sheets')
