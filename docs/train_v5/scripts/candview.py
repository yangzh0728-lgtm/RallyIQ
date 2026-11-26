import sys; sys.path.insert(0,'/Users/mac/Desktop/RallyIQ/train_v5/scripts'); sys.path.insert(0,'/Users/mac/Desktop/RallyIQ/train_v4/scripts')
import rt,replay,sheets,cv2,numpy as np
from rallyiq.perception.players.filtering import build_candidates
from rallyiq.perception.target_court import SegmentRegions
out=[];V={}
for a in sys.argv[2:]:
    k,t=a.split('@');t=float(t)
    if k not in V: V[k]=(sheets.Video(k),replay.load(k),replay.calib(k))
    v,by,cal=V[k]; f=round(t*v.fps)//2*2; im=v.frame(f/v.fps).copy(); reg=SegmentRegions(cal).for_frame(f)
    cs=build_candidates([d for d,_ in by.get(f,[])],np.array(cal.segment_for(f).homography),(1920,1080),None,reg)
    for c in cs:
        b=list(map(int,c.det.bbox)); cv2.rectangle(im,b[:2],b[2:],(255,200,0),3); cv2.putText(im,f"{c.player_score:.2f} {c.foot_court[0]:.1f},{c.foot_court[1]:.1f} {','.join(r[:4] for r in c.reasons)}",(b[0],b[1]-6),0,.8,(0,255,255),2)
    cv2.putText(im,a,(10,40),0,1,(255,255,255),2); out.append(cv2.resize(im,(960,540)))
while len(out)%2: out.append(np.zeros_like(out[0]))
cv2.imwrite(sys.argv[1],cv2.vconcat([cv2.hconcat(out[i:i+2]) for i in range(0,len(out),2)]))
