"""w45.py KEY T NAME: frames t-0.25..t+0.25 s, all detector candidates (red, score) + v5 track point (green)."""
import sys,json,cv2,numpy as np
k,t,name=sys.argv[1],float(sys.argv[2]),sys.argv[3]
C=json.load(open(f'/Users/mac/Desktop/RallyIQ/train_v5/inputs/cands/{k}.json'))['frames']
S={p['frame']:p for p in json.load(open(f'/Users/mac/Desktop/RallyIQ/train_v5/inputs_C/new_runs/{k}/shuttle.json'))['points']}
cap=cv2.VideoCapture(f'/Users/mac/Desktop/RallyIQ/train_v4/inputs/sources/{k}.mp4'); fps=cap.get(5)
rows=[]
for o in np.linspace(-.25,.25,6):
    f=int(round((t+o)*fps)); cap.set(1,f); ok,im=cap.read()
    for c in C.get(str(f),{}).get('candidates',[]):
        cv2.circle(im,(int(c[0]),int(c[1])),14,(0,0,255),2); cv2.putText(im,f'{c[2]:.2f}',(int(c[0])+14,int(c[1])),0,.8,(0,0,255),2)
    if f in S: cv2.circle(im,(int(S[f]['x']),int(S[f]['y'])),20,(0,255,0),3)
    im=cv2.resize(im,(960,540)); cv2.putText(im,f'{name} f={f} dt={o:+.2f}',(10,30),0,1,(255,255,255),3); rows.append(im)
g=np.vstack([np.hstack(rows[i:i+2]) for i in range(0,6,2)])
cv2.imwrite(f'/Users/mac/Desktop/RallyIQ/train_v6/review/w45/{name}.jpg',g,[cv2.IMWRITE_JPEG_QUALITY,75])
