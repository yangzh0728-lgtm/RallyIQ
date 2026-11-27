"""b_public.py WEIGHTS NAME: TrackNet public test (30 clips) and the 6 local ShuttleSet clips (test matches) with a
shuttle checkpoint: detect (peaks >= 0.1, top 3, whole clip) -> v6 tracker -> TrackNet shuttle F1 / ShuttleSet hit F1."""
import os,sys,json,subprocess
from pathlib import Path
R=Path('~/Desktop/RallyIQ/train_v6/repo').expanduser(); sys.path[:0]=[str(R),str(R/'scripts')]
from rallyiq.perception.shuttle.detector import HeatmapShuttleDetector,DetectorConfig
import cv2
W,name=sys.argv[1],sys.argv[2]; B6=Path('~/Desktop/RallyIQ/train_v6').expanduser(); O=B6/'ckpt/b'/f'pub_{name}'; O.mkdir(parents=True,exist_ok=True)
det=HeatmapShuttleDetector(Path(W).expanduser(),DetectorConfig(peak_threshold=0.1,top_k=3))
def cands(video,out):
    if out.exists(): return
    cap=cv2.VideoCapture(str(video)); n=int(cap.get(7)); fps=cap.get(5); cap.release()
    res,size=det.detect(Path(video),fps,n)
    out.write_text(json.dumps({'detector':det.name,'frame_size':list(size),'note':name,'frames':{str(f):{'candidates':[[round(x,2),round(y,2),round(s,4)] for x,y,s in c],'selected':-1} for f,c in res.items()}}))
PUB=Path('~/rallyiq-data/public').expanduser()
man=json.loads((PUB/'tracknet/manifest.json').read_text()); (O/'tn_cands').mkdir(exist_ok=True)
for cid in man:
    m,r=cid.split('_')[2],'_'.join(cid.split('_')[3:])
    cands(PUB/f'tracknet_ds/test/{m}/video/{r}.mp4',O/'tn_cands'/f'{cid}.json')
tn=subprocess.run([sys.executable,str(R/'scripts/tracknet_retrack.py'),'--candidates',str(O/'tn_cands'),'--out',str(O/'tn_preds')],capture_output=True,text=True).stdout.strip().splitlines()[-1]
s=json.loads(tn)['shuttle']['all']; print('TRACKNET',name,'f1 %.4f acc %.4f p %.4f r %.4f'%(s['f1'],s['accuracy'],s['precision'],s['recall']),flush=True)
# ShuttleSet local clips: artifacts symlinked, shuttle.json rebuilt from the new candidates with the v6 tracker
sys.path.insert(0,'/Users/mac/Desktop/RallyIQ/train_v5/scripts'); os.environ['RTREPO']=str(R)
import rt
SS=PUB/'shuttleset'; res=[{'clip_id':p.stem,'artifacts':str(SS/'data/videos'/json.loads(p.read_text())['source_video_id']/'artifacts')} for p in sorted((SS/'preds').glob('ss*.json')) if not p.name.startswith('._')]; D=O/'ss'; (D/'cands').mkdir(parents=True,exist_ok=True)
for e in res:
    cid=e['clip_id']; ad=Path(e['artifacts']); cands(SS/'clips'/f'{cid}.mp4',D/'cands'/f'{cid}.json')
    od=D/'data/videos'/ad.parent.name/'artifacts'; od.mkdir(parents=True,exist_ok=True)
    for n in os.listdir(ad):
        if n.startswith('.') or n.startswith('shuttle.json') or n.startswith('events') : continue
        if not (od/n).exists(): os.symlink(ad/n,od/n)
    if not (od/'shuttle.json').exists():
        os.symlink(ad/'shuttle.json',od/'shuttle.json')
    c=rt.Ctx(cid,str(od),str(D/'cands'/f'{cid}.json')); tr,_=c.run(); os.remove(od/'shuttle.json'); (od/'shuttle.json').write_text(tr.model_dump_json())
    e['artifacts']=str(od)
(D/'results.json').write_text(json.dumps(res))
for n in ('labels','manifest.json','preds'):
    if not (D/n).exists(): os.symlink(SS/n,D/n)
ev=subprocess.run([sys.executable,str(R/'scripts/learn_hits.py'),'eval','--data',str(D),'--model',str(R/'rallyiq/events/models/hit_scorer.json'),'--out-name','e'],capture_output=True,text=True)
print('\n'.join(l for l in ev.stdout.splitlines() if l.startswith('| rules') or l.startswith('| hit_scorer')),flush=True)
print(ev.stderr[-500:] if ev.returncode else '')
