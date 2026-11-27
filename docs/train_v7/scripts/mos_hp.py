"""mos_hp.py CANDS OUTPREFIX: crops (100x100 -> 150x150) around each estimate, 8x6 per mosaic, id printed."""
import json,sys,cv2,numpy as np,os
C=json.load(open(sys.argv[1])); pre=sys.argv[2]; os.makedirs(os.path.dirname(pre),exist_ok=True)
src=lambda k: next(s for s in (f'/Users/mac/Desktop/RallyIQ/train_v4/inputs/sources/{k}.mp4',f'/Users/mac/Desktop/RallyIQ/train_v5/inputs/sources/{k}.mp4',f'/Users/mac/Desktop/RallyIQ/train_v4/inputs/new_runs/{k}/source.mp4') if os.path.exists(s))
caps={}; tiles={}
for o in sorted(C,key=lambda o:(o['k'],o['frame'])):
    cap=caps.setdefault(o['k'],cv2.VideoCapture(src(o['k']))); cap.set(1,o['frame']); ok,im=cap.read()
    x,y=int(o['x']),int(o['y']); r=50; pad=cv2.copyMakeBorder(im,r,r,r,r,cv2.BORDER_CONSTANT)
    c=pad[y:y+2*r,x:x+2*r].copy(); c=cv2.resize(c,(150,150),interpolation=cv2.INTER_CUBIC)
    cv2.circle(c,(75,75),14,(0,0,255),1); cv2.putText(c,str(o['id']),(3,16),0,.5,(0,255,255),1); tiles[o['id']]=c
ids=sorted(tiles); 
for m in range(0,len(ids),48):
    g=[tiles[i] for i in ids[m:m+48]]; g+= [np.zeros_like(g[0])]*(48-len(g))
    cv2.imwrite(f'{pre}_{m//48:02d}.jpg',np.vstack([np.hstack(g[r*8:(r+1)*8]) for r in range(6)]),[cv2.IMWRITE_JPEG_QUALITY,85])
print(len(ids))
