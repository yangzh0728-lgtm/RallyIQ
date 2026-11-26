import sys; sys.path.insert(0,'/Users/mac/Desktop/RallyIQ/train_v4/scripts')
import sheets,numpy as np,cv2
v=sheets.Video('own_20260324_0241'); t=float(sys.argv[1]) if len(sys.argv)>1 else 140.4
im=v.frame(t).copy()
pts=[((845,805),(3.05,4.72)),((336,891),(.46,4.72)),((212,910),(0,4.72)),((1513,1060),(3.05,.76)),((1860,879),(6.1,.76)),((1822,897),(5.64,.76)),((189,813),(0,6.7))]
I=np.array([p for p,_ in pts],float);C=np.array([c for _,c in pts],float)
H,_=cv2.findHomography(C,I)
def P(x,y): q=H@[x,y,1]; return q[:2]/q[2]
for p,c in pts: print(p,P(*c).round(1))
np.save('/Users/mac/Desktop/RallyIQ/train_v5/inputs/H0241_140p4.npy',H)
w0=(H@[3.05,6.7,1])[2]
def line(p,q,c):
    L=[(H@[*(np.array(p)+(np.array(q)-p)*u),1]) for u in np.linspace(0,1,300)]
    for a,b in zip(L,L[1:]):
        if a[2]*w0>0 and b[2]*w0>0:
            a2,b2=a[:2]/a[2],b[:2]/b[2]
            if max(abs(a2).max(),abs(b2).max())<9000: cv2.line(im,tuple(map(int,a2)),tuple(map(int,b2)),c,2)
for p,q in [((0,0),(6.1,0)),((6.1,0),(6.1,13.4)),((6.1,13.4),(0,13.4)),((0,13.4),(0,0)),((.46,0),(.46,13.4)),((5.64,0),(5.64,13.4)),((0,4.72),(6.1,4.72)),((0,8.68),(6.1,8.68)),((0,.76),(6.1,.76)),((0,12.64),(6.1,12.64)),((3.05,0),(3.05,4.72)),((3.05,8.68),(3.05,13.4))]: line(p,q,(0,0,255))
line((0,6.7),(6.1,6.7),(255,0,255))
cv2.imwrite('/Users/mac/Desktop/RallyIQ/train_v5/review/fit0241.jpg',im)
