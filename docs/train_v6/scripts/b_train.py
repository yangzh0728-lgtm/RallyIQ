"""b_train.py OUT [--lr 1e-4 --epochs 8 --steps 300 --tn-share 0.33 --hp-weight 4] : fine-tune heatmap.pt on own
flights (+hard positives) with TrackNet rehearsal; per-epoch val on own easy / own hard / TrackNet guard frames.
Also `--eval WEIGHTS...` to only score checkpoints on the val sets."""
import sys,os,json,math,random,time,argparse
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path('~/Desktop/RallyIQ/train_v6/repo').expanduser()))
import cv2,torch
from rallyiq.perception.shuttle import training as T
from rallyiq.perception.shuttle.model import INPUT_WIDTH,INPUT_HEIGHT
B6=Path('~/Desktop/RallyIQ/train_v6').expanduser(); BASE=Path('~/RallyIQ/weights/shuttle/heatmap.pt').expanduser()
ap=argparse.ArgumentParser(); ap.add_argument('out',nargs='?'); ap.add_argument('--lr',type=float,default=1e-4); ap.add_argument('--epochs',type=int,default=8)
ap.add_argument('--steps',type=int,default=300); ap.add_argument('--tn-share',type=float,default=0.33); ap.add_argument('--hp-weight',type=float,default=4.0)
ap.add_argument('--eval',nargs='*'); ap.add_argument('--bs',type=int,default=4)
a=ap.parse_args()
S=json.load(open(B6/'ckpt/b/spec.json'))
MED={}
def video_median(path):
    p=B6/'ckpt/b'/('med_'+Path(path).stem+'.npy')
    if p.exists(): return np.load(p)
    cap=cv2.VideoCapture(path); n=int(cap.get(7)); fr=[]
    for i in np.unique(np.linspace(0,n-1,120).astype(int)):
        cap.set(1,int(i)); ok,im=cap.read()
        if ok: fr.append(cv2.resize(im,(INPUT_WIDTH,INPUT_HEIGHT),interpolation=cv2.INTER_AREA)[:,:,::-1])
    m=np.median(np.stack(fr),axis=0).astype(np.uint8); np.save(p,m); return m
_init=T.ClipFrames.__init__
def init(self,clip,L,use_bg,target_fps=30.0):
    _init(self,clip,L,use_bg,target_fps)
    if use_bg and str(clip.video) in MED: self.median=MED[str(clip.video)]
T.ClipFrames.__init__=init
def clips(items,own):
    out=[]
    for x in items:
        if own and x['video'] not in MED: MED[x['video']]=video_median(x['video'])
        lab={int(f):(None if v is None else tuple(v)) for f,v in x['labels'].items()}
        out.append(T.TrainClip(x['id'],Path(x['video']),x['fps'],lab))
    return out
cfg=T.TrainConfig(epochs=a.epochs,steps_per_epoch=a.steps,batch_size=a.bs,lr=a.lr,select_at=0.2)
device=T.pick_device(None) if hasattr(T,'pick_device') else None
from rallyiq.perception.shuttle.detector import pick_device; device=pick_device(None)
t0=time.time()
own_val=[T.ClipFrames(c,8,True) for c in clips(S['own']['val'],True)]
tn_val=[T.ClipFrames(c,8,True) for c in clips(S['tn']['val'],False)]
HPV={(Path(x['video']).stem,f) for x in S['own']['val'] for f in x['hp']}
print('val decoded',sum(len(d.frames) for d in own_val+tn_val),'frames',round(time.time()-t0),'s',flush=True)
def metrics(net):
    out={}
    for name,data in (('own',own_val),('tn',tn_val)):
        pk={}; T.evaluate(net,data,cfg,device,peaks=pk)
        sub={'all':lambda s,f:True,'hard':lambda s,f:(s,f) in HPV,'easy':lambda s,f:(s,f) not in HPV} if name=='own' else {'all':lambda s,f:True}
        for sn,sel in sub.items():
            for thr in (0.2,0.3):
                c={'tp':0,'fp':0,'fn':0}
                for d in data:
                    st=d.clip.video.stem
                    for f,truth in d.clip.labels.items():
                        if not sel(st,f): continue
                        p=pk.get(st,{}).get(f); fired=p is not None and p[2]>=thr; tol=cfg.tol_px*d.height/1080
                        if truth is None: c['fp']+=fired
                        elif not fired: c['fn']+=1
                        elif math.hypot(p[0]-truth[0],p[1]-truth[1])<=tol: c['tp']+=1
                        else: c['fp']+=1; c['fn']+=1
                pr=c['tp']/max(c['tp']+c['fp'],1); rc=c['tp']/max(c['tp']+c['fn'],1)
                out[f'{name}_{sn}@{thr}']={'p':round(pr,4),'r':round(rc,4),'f1':round(2*pr*rc/max(pr+rc,1e-9),4),**c}
    return out
def show(m): return ' '.join(f"{k}={v['f1']:.4f}(r{v['r']:.3f})" for k,v in m.items() if k.endswith('@0.2'))
if a.eval is not None:
    for w in a.eval or [str(BASE)]:
        net,_=T.load_model(Path(w).expanduser(),device); m=metrics(net); print(w,show(m),flush=True)
        json.dump(m,open(B6/'ckpt/b'/(Path(w).stem+'.val.json'),'w'))
    sys.exit()
own_tr=[T.ClipFrames(c,8,True) for c in clips(S['own']['train'],True)]
tn_tr=[T.ClipFrames(c,8,True) for c in clips(S['tn']['train'],False)]
HPT={(x['id'],f) for x in S['own']['train'] for f in x['hp']}
print('train decoded',sum(len(d.frames) for d in own_tr+tn_tr),'frames',round(time.time()-t0),'s',flush=True)
pool,w=[],[]
for d in own_tr:
    for s in d.starts:
        hard=any((d.clip.clip_id,s+k*d.step) in HPT for k in range(8)); pool.append((d,s)); w.append(('own',a.hp_weight if hard else 1.0))
for d in tn_tr:
    for s in d.starts: pool.append((d,s)); w.append(('tn',1.0))
wo=sum(x for g,x in w if g=='own'); wt=sum(x for g,x in w if g=='tn')
W=[x*((1-a.tn_share)/wo if g=='own' else a.tn_share/wt) for g,x in w]
net,params=T.load_model(BASE,device)
base_m=metrics(net); print('base',show(base_m),flush=True)
hist=[{'epoch':0,**base_m}]; best=(base_m['own_all@0.2']['f1'],0); best_state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
opt=torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],lr=a.lr,weight_decay=cfg.weight_decay)
rng=random.Random(0)
for ep in range(1,a.epochs+1):
    net.train(); T._set_bn_eval(net); tl=[]
    for _ in range(a.steps):
        xs,hms,ms=[],[],[]
        for d,s in rng.choices(pool,weights=W,k=a.bs):
            flip=rng.random()<.5; gain,bias=1+rng.uniform(-.2,.2),rng.uniform(-.1,.1)
            idx,fr=d.window(s); xs.append(T.make_input(fr,d.median,flip,gain,bias)); hm,m=d.target(idx,flip); hms.append(hm); ms.append(m)
        y=net(torch.from_numpy(np.stack(xs)).to(device))
        loss=T.wbce(y,torch.from_numpy(np.stack(hms)).to(device),torch.from_numpy(np.stack(ms)).to(device))
        opt.zero_grad(); loss.backward(); opt.step(); tl.append(float(loss.detach()))
    m=metrics(net); hist.append({'epoch':ep,'train_loss':float(np.mean(tl)),**m})
    ok=m['tn_all@0.2']['f1']>=base_m['tn_all@0.2']['f1']-0.005
    print(f'epoch {ep} loss {np.mean(tl):.2e}',show(m),'tn_ok' if ok else 'TN DROP',round(time.time()-t0),'s',flush=True)
    if ok and m['own_all@0.2']['f1']>best[0]: best=(m['own_all@0.2']['f1'],ep); best_state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
    json.dump(hist,open(str(a.out)+'.hist.json','w'))
torch.save({'model':best_state,'param_dict':params,'finetune':{'best_epoch':best[1],'hist':hist,'args':vars(a)}},a.out)
print('saved epoch',best[1],'->',a.out)
