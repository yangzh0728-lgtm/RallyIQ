"""rowstrip.py ROWS.json OUTDIR INPUTS: per row 3 panels (-0.2/0/+0.2 s) cropped around players + track from INPUTS -> 4 rows per image."""
import sys,os,json,cv2,numpy as np
sys.path.insert(0,'/Users/mac/Desktop/RallyIQ/train_v5/scripts'); import rt
OFFS=[float(x) for x in os.environ.get('OFFS','-.2,0,.2').split(',')]
rows=json.load(open(sys.argv[1])); out=sys.argv[2]; inp=sys.argv[3]; os.makedirs(out,exist_ok=True)
IDV={l.split()[0]:l.split()[1] for l in open('/Users/mac/Desktop/RallyIQ/train_v5/ids.txt')}
def src(k): return next(s for s in (f'{rt.H}/train_v4/inputs/sources/{k}.mp4',f'{rt.B}/inputs/sources/{k}.mp4',f'{rt.H}/train_v4/inputs/new_runs/{k}/source.mp4',os.path.expanduser(f'~/rallyiq-data/runs/own-v12/data/videos/{IDV[k]}/source.mp4')) if os.path.exists(s))
cache={}; tiles=[]
for i,(k,t,side,*_) in enumerate(rows):
    if k not in cache:
        a=f'{inp}/new_runs/{k}'; cache[k]=(json.loads(rt.rd(a,'probe'))['fps'],{p['frame']:p for p in json.load(open(f'{a}/shuttle.json'))['points']},{f['frame']:f['players'] for f in json.loads(rt.rd(a,'players'))['frames']},cv2.VideoCapture(src(k)))
    fps,S,PL,cap=cache[k]; f0=round(t*fps); ps=[]
    bs=[p['bbox'] for d in range(-4,5) for p in PL.get(f0+d,[])]
    xs=[b[0] for b in bs]+[b[2] for b in bs]; ys=[b[1] for b in bs]+[b[3] for b in bs]
    x0,x1=(max(min(xs)-150,0),min(max(xs)+150,1920)) if xs else (0,1920); y0,y1=(max(min(ys)-250,0),min(max(ys)+40,1080)) if ys else (0,1080)
    x0,x1,y0,y1=map(int,(x0,x1,y0,y1))
    for o in OFFS:
        f=round((t+o)*fps); cap.set(1,f); ok,im=cap.read()
        for d in (0,-1,1):
            if f+d in PL:
                for p in PL[f+d]:
                    b=list(map(int,p['bbox'])); cv2.rectangle(im,b[:2],b[2:],(0,200,0) if p.get('side')=='near' else (255,160,0),2)
                break
        for g in range(f-int(.25*fps),f+1):
            if g in S: cv2.circle(im,(int(S[g]['x']),int(S[g]['y'])),12 if g==f else 4,(0,255,255),2 if g==f else -1)
        c=cv2.resize(im[y0:y1,x0:x1],(560,int(560*(y1-y0)/max(x1-x0,1))))
        c=cv2.resize(c,(480,330)); cv2.putText(c,f"#{i} {k[-7:]} {t:.2f} {_[0] if _ else side} {o:+.1f}",(6,24),0,.7,(255,255,255),2); ps.append(c)
    tiles.append(np.hstack(ps))
for s in range(0,len(tiles),4): cv2.imwrite(f'{out}/r_{s//4:02d}.jpg',np.vstack(tiles[s:s+4]),[cv2.IMWRITE_JPEG_QUALITY,75])
print(len(tiles))
