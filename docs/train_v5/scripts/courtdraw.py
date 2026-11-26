import sys; sys.path.insert(0,'/Users/mac/Desktop/RallyIQ/train_v5/scripts'); sys.path.insert(0,'/Users/mac/Desktop/RallyIQ/train_v4/scripts')
import rt,cv2,numpy as np,sheets
out=[]
for a in sys.argv[2:]:
    k,t=a.split('@');t=float(t);v=sheets.Video(k);f=round(t*v.fps);im=v.frame(t).copy()
    import os;fp=f'{rt.B}/inputs/fixcal/{k}.json';cal=rt.CourtCalibration.model_validate_json(open(fp).read() if os.path.exists(fp) else rt.rd(rt.art(k),'calibration'));s=cal.segment_for(f);H=np.linalg.inv(np.array(s.homography))
    def P(x,y): q=H@[x,y,1]; return q[:2]/q[2],q[2]
    _,w0=P(3.05,6.7)
    def line(p,q,c):
        pts=[P(*(np.array(p)+(np.array(q)-p)*u)) for u in np.linspace(0,1,200)]
        for (a,wa),(b,wb) in zip(pts,pts[1:]):
            if wa*w0>0 and wb*w0>0 and max(abs(a).max(),abs(b).max())<9000: cv2.line(im,tuple(map(int,a)),tuple(map(int,b)),c,3)
    for p,q in [((0,0),(6.1,0)),((6.1,0),(6.1,13.4)),((6.1,13.4),(0,13.4)),((0,13.4),(0,0)),((.46,0),(.46,13.4)),((5.64,0),(5.64,13.4)),((0,4.72),(6.1,4.72)),((0,8.68),(6.1,8.68)),((0,.76),(6.1,.76)),((0,12.64),(6.1,12.64))]: line(p,q,(0,0,255))
    line((0,6.7),(6.1,6.7),(255,0,255))
    for p in v.players(f):
        if not p['player_id']: continue
        b=list(map(int,p['bbox']));cv2.rectangle(im,b[:2],b[2:],(255,200,0),3);cv2.putText(im,f"{p['player_id']} {p['foot_court'][0]:.1f},{p['foot_court'][1]:.1f}",(b[0],b[1]-8),0,.9,(0,255,255),2);cv2.circle(im,tuple(map(int,p['foot_px'])),7,(0,255,0),-1)
    cv2.putText(im,f'{a} {s.method} {s.notes[:1]}',(10,40),0,1,(255,255,255),2); out.append(cv2.resize(im,(960,540)))
while len(out)%2: out.append(np.zeros_like(out[0]))
cv2.imwrite(sys.argv[1],cv2.vconcat([cv2.hconcat(out[i:i+2]) for i in range(0,len(out),2)]))
