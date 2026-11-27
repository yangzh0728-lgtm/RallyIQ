"""strip.py OUT KEY:T[:TAG] ...: 3 frames (-0.15, 0, +0.15 s) with v5 player boxes + ids and the shuttle track."""
import sys,os,json,cv2,numpy as np
sys.path.insert(0,'/Users/mac/Desktop/RallyIQ/train_v5/scripts'); import rt
out=sys.argv[1]; os.makedirs(out,exist_ok=True)
PL=os.environ.get('PLAYERS',f'{rt.B}/players_v5'); SH=os.environ.get('SHUTTLE',f'{rt.B}/inputs_C/new_runs')
for a in sys.argv[2:]:
    k,t,*tag=a.split(':'); t=float(t); fps=float(rt.IDS[k][2]); dt=int(round(fps*.15))
    P=json.load(open(f'{PL}/{k}.json')); PB={f['frame']:f['players'] for f in P['frames']}
    S={p['frame']:p for p in json.load(open(f'{SH}/{k}/shuttle.json'))['points']}
    src=next(s for s in (f'{rt.H}/train_v4/inputs/sources/{k}.mp4',f'{rt.B}/inputs/sources/{k}.mp4',f'{rt.H}/train_v4/inputs/new_runs/{k}/source.mp4') if os.path.exists(s))
    cap=cv2.VideoCapture(src); f0=round(t*fps); rows=[]
    for o in (-dt,0,dt):
        g=f0+o; cap.set(1,g); ok,im=cap.read()
        pf=PB.get(g) or PB.get(g-1) or PB.get(g+1) or []
        for p in pf:
            b=[int(v) for v in p['bbox']]; col=(0,200,0) if p['side']=='near' else (255,160,0)
            cv2.rectangle(im,(b[0],b[1]),(b[2],b[3]),col,2); cv2.putText(im,p['player_id']+p['side'][0],(b[0],b[1]-8),0,1.1,col,3)
        for h in range(g-3,g+4):
            if h in S: cv2.circle(im,(int(S[h]['x']),int(S[h]['y'])),10 if h==g else 5,(0,255,255),2)
        im=cv2.resize(im,(960,540)); cv2.putText(im,f"{k[-7:]} t={t:.2f} {o:+d}f {' '.join(tag)}",(8,30),0,.9,(255,255,255),3); rows.append(im)
    cv2.imwrite(f"{out}/{k[-7:]}_{t:.2f}.jpg",np.vstack(rows),[cv2.IMWRITE_JPEG_QUALITY,75])
