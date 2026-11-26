"""pv.py out key@t ... : slots of players_v5 (or PD) drawn"""
import sys,os; sys.path.insert(0,'/Users/mac/Desktop/RallyIQ/train_v5/scripts'); sys.path.insert(0,'/Users/mac/Desktop/RallyIQ/train_v4/scripts')
import rt,sheets5 as sheets,cv2,numpy as np
P={};V={};out=[]
for a in sys.argv[2:]:
    k,t=a.split('@');t=float(t)
    if k not in P:
        pt=rt.PlayerTracks.model_validate_json(open(f'{rt.B}/players_v5/{k}.json').read());P[k]={f.frame:f.players for f in pt.frames};V[k]=sheets.Video(k)
    v=V[k];f=round(t*v.fps)//2*2;im=v.frame(f/v.fps).copy()
    for p in P[k].get(f,[]):
        b=list(map(int,p.bbox));cv2.rectangle(im,b[:2],b[2:],(0,200,255),4);cv2.putText(im,f'{p.player_id}{p.side[0]} {p.foot_court[0]:.1f},{p.foot_court[1]:.1f}',(b[0],b[1]-8),0,1,(0,255,255),3)
    for p in v.players(f):
        if p['player_id']: b=list(map(int,p['bbox']));cv2.rectangle(im,b[:2],b[2:],(255,120,0),1);cv2.putText(im,p['player_id'],(b[0],b[3]+18),0,.7,(255,120,0),2)
    cv2.putText(im,a,(10,40),0,1,(255,255,255),2);out.append(cv2.resize(im,(960,540)))
while len(out)%2: out.append(np.zeros_like(out[0]))
cv2.imwrite(sys.argv[1],cv2.vconcat([cv2.hconcat(out[i:i+2]) for i in range(0,len(out),2)]),[1,85])
