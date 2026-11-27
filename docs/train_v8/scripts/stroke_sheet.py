"""stroke_sheet.py: one row per fresh labelled hit: contact frame with the whole flight to the next hit drawn (blue->red
over time), plus frames at +0.3 / +0.6 s and at the next hit. -> review/strokes/s_NN.jpg (4 rows), ckpt/stroke_items.json"""
import json,cv2,numpy as np,os
B6='/Users/mac/Desktop/RallyIQ/train_v6'; out='review/strokes2'; os.makedirs(out,exist_ok=True); items=[]
for k in ('own_20260529_0005','own_20260529_0007','own_20260529_0009'):
    L=[x for x in ([y.strip() for y in l.split('|')] for l in open(f'{B6}/labels/d/{k}.txt') if l[0].isdigit()) if x[1]=='hit']
    ts=sorted(float(x[2]) for x in L)
    for x in sorted(L,key=lambda x:float(x[2])):
        t=float(x[2]); nx=[u for u in ts if u>t+.15]; n=nx[0] if nx and nx[0]-t<3 else t+1.6
        items.append({'k':k,'t':t,'side':x[3],'next':n,'why':x[4]})
json.dump(items,open('ckpt/stroke_items.json','w'))
caps={}; rows=[]
for i,it in enumerate(items):
    a=f'{B6}/inputs/d_run/new_runs/{it["k"]}'
    if it['k'] not in caps: caps[it['k']]=(cv2.VideoCapture(a+'/source.mp4'),{p['frame']:p for p in json.load(open(a+'/shuttle.json'))['points']})
    cap,S=caps[it['k']]; fps=cap.get(5); f0=round(it['t']*fps); f1=round(it['next']*fps)
    path=[(S[g]['x'],S[g]['y'],(g-f0)/max(f1-f0,1)) for g in range(f0,f1+1) if g in S]
    ps=[]
    for j,tt in enumerate((it['t'],it['next'])):
        cap.set(1,round(tt*fps)); ok,im=cap.read()
        for x,y,u in path: cv2.circle(im,(int(x),int(y)),5,(int(255*(1-u)),0,int(255*u)),-1)
        im=cv2.resize(im[100:820,250:1530],(800,450)); cv2.putText(im,f"#{i} {it['k'][-4:]} {it['t']:.2f} {it['side']} {['hit','next'][j]}",(6,24),0,.65,(255,255,255),2); ps.append(im)
    rows.append(np.hstack(ps))
for s in range(0,len(rows),5): cv2.imwrite(f'{out}/s_{s//5:02d}.jpg',np.vstack(rows[s:s+5]),[cv2.IMWRITE_JPEG_QUALITY,72])
print(len(items))
