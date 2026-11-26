"""Corrected courts for 0241 and 0245 -> inputs/fixcal/<key>.json (CourtCalibration)."""
import sys; sys.path.insert(0,'/Users/mac/Desktop/RallyIQ/train_v5/scripts')
import rt,numpy as np,cv2,json,os
os.makedirs(rt.B+'/inputs/fixcal',exist_ok=True)
def seg_update(s,Hci,note):
    d=s.model_dump(); Hi=np.linalg.inv(Hci); Hi/=Hi[2,2]
    d['homography']=Hi.tolist()
    c=[[0,0],[6.1,0],[6.1,13.4],[0,13.4]]; im=[(lambda q:(q[:2]/q[2]).tolist())(Hci@[*p,1]) for p in c]
    d['image_corners']=im; d['keypoints']=[{**k,'image':i} for k,i in zip(d['keypoints'],im)] if len(d['keypoints'])==4 else d['keypoints']
    d['notes']=[note]; return d
def save(k,segs,cal):
    d=cal.model_dump(); d['segments']=segs; n=rt.CourtCalibration.model_validate(d)
    open(f'{rt.B}/inputs/fixcal/{k}.json','w').write(n.model_dump_json())
k='own_20260324_0241';cal=rt.CourtCalibration.model_validate_json(rt.rd(rt.art(k),'calibration'));fps=59.94005994
ref=np.linalg.inv(np.array(cal.segment_for(round(140.4*fps)).homography)); new=np.load(rt.B+'/inputs/H0241_140p4.npy')
D=new@np.linalg.inv(ref)  # wrong: need image motion -> see below
segs=[]
for s in cal.segments:
    old=np.linalg.inv(np.array(s.homography)); M=old@np.linalg.inv(ref)  # image motion ref->seg (both from the same propagated court)
    segs.append(seg_update(s,M@new,'court re-fitted from the painted lines at 140.4 s (train_v5); pose propagated as before'))
save(k,segs,cal)
k='own_20260324_0245';cal=rt.CourtCalibration.model_validate_json(rt.rd(rt.art(k),'calibration'))
pts=[((840,675),(3.05,4.72)),((357,760),(.46,4.72)),((1503.5,887.5),(3.05,.76)),((1788.5,976),(3.05,0)),((1794,729),(6.1,.76)),((1768,743.5),(5.64,.76)),((237,780),(0,4.72))]
H,_=cv2.findHomography(np.array([c for _,c in pts],float),np.array([p for p,_ in pts],float))
for p,c in pts: q=H@[*c,1]; print(p,(q[:2]/q[2]).round(1))
save(k,[seg_update(s,H,'court fitted from the painted lines at 326 s (train_v5); was a net-post perspective model') for s in cal.segments],cal)
