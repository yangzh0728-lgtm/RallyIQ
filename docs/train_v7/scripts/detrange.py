"""detrange.py WEIGHTS NAME [keys...]: re-run the shuttle detector inside the labelled windows (+-3 s) of each video with
the per-video background median, peaks >= 0.1 (top 3) -> ckpt/b/cands_NAME/<key>.json (old candidates outside the windows)."""
import sys,os,json,time
from pathlib import Path
import numpy as np, cv2, torch
sys.path.insert(0,str(Path('~/Desktop/RallyIQ/train_v6/repo').expanduser())); sys.path.insert(0,os.path.dirname(__file__))
from rallyiq.perception.shuttle.detector import HeatmapShuttleDetector,DetectorConfig
from rallyiq.perception.shuttle.heatmap import extract_peaks
from rallyiq.perception.shuttle.model import INPUT_WIDTH,INPUT_HEIGHT
B5='/Users/mac/Desktop/RallyIQ/train_v5'; B6=Path('/Users/mac/Desktop/RallyIQ/train_v6')
W,name=sys.argv[1],sys.argv[2]; keys=sys.argv[3:]
L4=json.load(open('/Users/mac/Desktop/RallyIQ/train_v5/labels/amateur_v4.json'))['windows']
Q=json.load(open('/Users/mac/Desktop/RallyIQ/train_v4/inputs/hits_truth.json'))['windows']
WIN={**{k:v for k,v in Q.items()},**L4}
DD=B6/'inputs/d'
for p in (B6/'ckpt/d/v6').glob('*.json'): WIN[p.stem]=json.load(open(p))['windows']
from collections import defaultdict
IDV=defaultdict(str,{l.split()[0]:l.split()[1] for l in open('/Users/mac/Desktop/RallyIQ/train_v5/ids.txt')})
src=lambda k: next(s for s in (str(DD/k/'source.mp4'),os.path.expanduser(f'~/rallyiq-data/runs/own-v12/data/videos/{IDV[k]}/source.mp4'),f'/Users/mac/Desktop/RallyIQ/train_v4/inputs/sources/{k}.mp4',f'{B5}/inputs/sources/{k}.mp4',f'/Users/mac/Desktop/RallyIQ/train_v4/inputs/new_runs/{k}/source.mp4') if os.path.exists(s))
IDS5={l.split()[0]:l.split()[-1].replace('~',os.path.expanduser('~')) for l in open('/Users/mac/Desktop/RallyIQ/train_v5/ids.txt')}
def rt_art(k): return IDS5[k]
import gzip
def rd(d,n):
    for p in (f'{d}/{n}.json',f'{d}/{n}.json.gz'):
        if os.path.exists(p): return gzip.open(p,'rt').read() if p.endswith('gz') else open(p).read()
det=HeatmapShuttleDetector(Path(W).expanduser(),DetectorConfig(peak_threshold=0.1,top_k=3))
out_dir=B6/f'ckpt/b/cands_{name}'; out_dir.mkdir(parents=True,exist_ok=True)
def median(k):
    """HWC RGB uint8 at network size, 120 frames across the video (same file and convention as b_train.py)."""
    p=B6/'ckpt/b'/f'med_{k}.npy'
    if p.exists(): return np.load(p)
    cap=cv2.VideoCapture(src(k)); n=int(cap.get(7)); fr=[]
    for i in np.unique(np.linspace(0,n-1,120).astype(int)):
        cap.set(1,int(i)); ok,im=cap.read()
        if ok: fr.append(cv2.resize(im,(INPUT_WIDTH,INPUT_HEIGHT),interpolation=cv2.INTER_AREA)[:,:,::-1])
    m=np.median(np.stack(fr),axis=0).astype(np.uint8); np.save(p,m); return m
def run(k):
    fps=json.load(open(DD/k/'probe.json' if (DD/k).exists() else f'{B5}/inputs_C/new_runs/{k}/probe.json'))['fps']; stride=max(1,round(fps/30)); Lq=det.seq_len
    # production background: one median per calibration segment (stage.segments_for, the run's own calibration)
    from rallyiq.perception.shuttle.stage import segments_for
    from rallyiq.contracts import CourtCalibration,VideoMeta
    ad=DD/k if (DD/k).exists() else Path(rt_art(k))
    meta=VideoMeta.model_validate_json(rd(ad,'probe')); cj=rd(ad,'calibration')
    segs=segments_for(meta,CourtCalibration.model_validate_json(cj) if cj else None,'off')
    meds=[torch.from_numpy(m).to(det.device) for m in det._medians(Path(src(k)),segs)]
    def bgf(f): return meds[next((i for i,s in enumerate(segs) if s.start<=f<=s.end),len(segs)-1)]
    old=json.load(open(DD/k/'shuttle_candidates.json' if (DD/k).exists() else f'{B5}/inputs/cands/{k}.json'))
    ranges=[]
    for a,b in sorted(WIN[k]):
        s,e=max(0,int((a-3)*fps)),int((b+3)*fps)
        if ranges and s<=ranges[-1][1]: ranges[-1][1]=max(ranges[-1][1],e)
        else: ranges.append([s,e])
    cap=cv2.VideoCapture(src(k)); W0=H0=None; res={}
    for s,e in ranges:
        cap.set(1,s); frames=[]
        for f in range(s,e+1):
            ok,im=cap.read()
            if not ok: break
            H0,W0=im.shape[:2]; frames.append(det._prep_u8(im,None))
        acc={}; accw={}
        for o in range(stride):
            idx=list(range(o,len(frames),stride))
            if not idx: continue
            starts=list(range(0,max(1,len(idx)-Lq+1)))+[len(idx)-Lq+j for j in range(1,Lq) if len(idx)-Lq+j>0]
            wins=[]
            for st in starts:
                w=idx[st:st+Lq]; n_real=len(w); w=w+[w[-1]]*(Lq-len(w)); wins.append((w,n_real))
            for b0 in range(0,len(wins),8):
                bt=wins[b0:b0+8]
                x=torch.from_numpy(np.stack([np.concatenate([frames[i] for i in w],0) for w,_ in bt])).to(det.device).float().div_(255.)
                x=torch.cat([torch.stack([bgf(s+w[0]) for w,_ in bt]),x],1)
                with torch.no_grad(): y=det.net(x).float().cpu().numpy()
                for (w,n_real),hms in zip(bt,y):
                    for pos,i in enumerate(w[:n_real]):
                        wt=float(det.weights_vec[pos]); acc[i]=acc.get(i,0)+wt*hms[pos]; accw[i]=accw.get(i,0)+wt
        sx,sy=W0/INPUT_WIDTH,H0/INPUT_HEIGHT
        for i in acc:
            pk=extract_peaks(acc[i]/max(accw[i],1e-6),0.1,3)
            res[s+i]=[[round((x+.5)*sx-.5,2),round((y+.5)*sy-.5,2),round(float(sc),4)] for x,y,sc in pk]
    fr=dict(old['frames'])
    for f,c in res.items(): fr[str(f)]={'candidates':c,'selected':-1}
    json.dump({**old,'detector':f'{name}','frames':fr},open(out_dir/f'{k}.json','w'))
    return len(res)
for k in keys:
    t0=time.time(); n=run(k); print(k,n,'frames',round(time.time()-t0),'s',flush=True)
