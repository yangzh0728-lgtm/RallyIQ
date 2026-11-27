"""n_sheet.py: every am7 labelled hit whose stroke is not sure -> review/n/s_NN.jpg (6 rows: contact | next hit, flight blue->red,
title = id, video, t, side, current label) + ckpt/n_items.json"""
import json,cv2,numpy as np,os,sys
sys.path.insert(0,'/Users/mac/Desktop/RallyIQ/train_v5/scripts'); import rt
L=json.load(open('labels/amateur_v6.json')); out='review/n'; os.makedirs(out,exist_ok=True); items=[]
for k,hs in sorted(L['hits'].items()):
    pos=sorted([h for h in hs if h['status'] in('reviewed_sure','reviewed_unsure')],key=lambda h:h['t'])
    ts=[h['t'] for h in pos]
    for h in pos:
        if h.get('stroke_sure') and h.get('stroke'): continue
        nx=[u for u in ts if u>h['t']+.15]; n=nx[0] if nx and nx[0]-h['t']<3 else h['t']+1.6
        items.append({'k':k,'t':h['t'],'side':h.get('side'),'next':n,'cur':h.get('stroke')})
json.dump(items,open('ckpt/n_items.json','w'))
IDV={l.split()[0]:l.split()[1] for l in open('/Users/mac/Desktop/RallyIQ/train_v5/ids.txt')}
src=lambda k: next(s for s in (f'{rt.H}/train_v4/inputs/sources/{k}.mp4',f'{rt.H}/train_v4/inputs/new_runs/{k}/source.mp4',os.path.expanduser(f'~/rallyiq-data/runs/own-v12/data/videos/{IDV[k]}/source.mp4')) if os.path.exists(s))
caps={}; rows=[]
for i,it in enumerate(items):
    k=it['k']; a=f'/Users/mac/Desktop/RallyIQ/train_v5/inputs_C/new_runs/{k}'
    if k not in caps: caps[k]=(cv2.VideoCapture(src(k)),{p['frame']:p for p in json.load(open(a+'/shuttle.json'))['points']})
    cap,S=caps[k]; fps=cap.get(5); f0=round(it['t']*fps); f1=round(it['next']*fps)
    path=[(S[g]['x'],S[g]['y'],(g-f0)/max(f1-f0,1)) for g in range(f0,f1+1) if g in S]
    ps=[]
    for j,tt in enumerate((it['t'],it['next'])):
        cap.set(1,round(tt*fps)); ok,im=cap.read()
        for x,y,u in path: cv2.circle(im,(int(x),int(y)),5,(int(255*(1-u)),0,int(255*u)),-1)
        im=cv2.resize(im,(800,450)); cv2.putText(im,f"#{i} {k[-7:]} {it['t']:.2f} {it['side']} cur={it['cur']} {['hit','next'][j]}",(6,26),0,.7,(255,255,255),2); ps.append(im)
    rows.append(np.hstack(ps))
for s in range(0,len(rows),6): cv2.imwrite(f'{out}/s_{s//6:02d}.jpg',np.vstack(rows[s:s+6]),[cv2.IMWRITE_JPEG_QUALITY,70])
print(len(items))
