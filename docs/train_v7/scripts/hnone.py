"""hnone.py: for labelled doubles hits with no hitter id, a 3-frame strip (-dt, 0, +dt) of the hit with raw detections
numbered (frame-0 numbering) -> review/hl/none_<key>_<n>.jpg"""
import sys,os,json,cv2,numpy as np
sys.path.insert(0,'/Users/mac/Desktop/RallyIQ/train_v5/scripts'); import rt
OUT=os.path.expanduser('~/Desktop/RallyIQ/train_v6/review/hl')
for k in sys.argv[1:]:
    M=json.load(open(f'{OUT}/{k}.json')); fps=float(rt.IDS[k][2]); dt=int(round(fps*0.15)); dt-=dt%2
    d=np.load(f'{rt.B}/dumps/{k}.npz'); fr=d['frame']; bb=d['bbox']; sc=d['score']
    S=json.load(open(f'{rt.B}/inputs_C/new_runs/{k}/shuttle.json')); SP={p['frame']:p for p in S['points']}
    cap=cv2.VideoCapture(f'{rt.H}/train_v4/inputs/sources/{k}.mp4')
    for m in M:
        if m['label'] is not None: continue
        rows=[]
        for o in (-dt,0,dt):
            g=m['frame']+o; cap.set(cv2.CAP_PROP_POS_FRAMES,g); ok,img=cap.read()
            for i in np.where(fr==g)[0]:
                b=[int(v) for v in bb[i]]; cv2.rectangle(img,(b[0],b[1]),(b[2],b[3]),(0,0,255),1)
            if o==0:
                for c in m['cand']:
                    b=[int(v) for v in c['bbox']]; col=(0,200,0) if c['pid'] else (0,0,255)
                    cv2.rectangle(img,(b[0],b[1]),(b[2],b[3]),col,2); cv2.putText(img,f"{c['i']}{(':'+c['pid']) if c['pid'] else ''} .{int(c['score']*100)}",(b[0],max(b[1]-6,12)),cv2.FONT_HERSHEY_SIMPLEX,.8,col,2)
            for h in range(g-2,g+3):
                if h in SP: cv2.circle(img,(int(SP[h]['x']),int(SP[h]['y'])),8,(0,255,255),2)
            y0,y1=(0,1080)
            c=cv2.resize(img[y0:y1],(960,int(960*(y1-y0)/1920))); cv2.putText(c,f"{k[-7:]} #{m['n']} t={m['t']:.2f} {m['side']} dt={o}",(8,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),3); rows.append(c)
        cv2.imwrite(f"{OUT}/none_{k}_{m['n']:02d}.jpg",np.vstack(rows),[cv2.IMWRITE_JPEG_QUALITY,75])
