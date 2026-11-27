"""r_train.py TAG [--seeds 3]: dilated TCN hit detector. Pretrain on ShuttleSet train, then for each of the 11 am7 videos a
model fine-tuned on ShuttleSet + the other 10 am7 videos (LOVO), plus one on all 11 (for qa6 / fresh / ShuttleSet test).
Writes per-clip hit probability and P(far) -> ckpt/r_<TAG>/probs.npz"""
import os,sys,glob,json,random,time,argparse
import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn
ap=argparse.ArgumentParser(); ap.add_argument('tag'); ap.add_argument('--seeds',type=int,default=3); ap.add_argument('--pre',type=int,default=3000)
ap.add_argument('--ft',type=int,default=1500); ap.add_argument('--ch',type=int,default=64); ap.add_argument('--own_w',type=float,default=3.0); ap.add_argument('--ssoof',type=int,default=0)
a=ap.parse_args()
B=os.path.expanduser('~/Desktop/RallyIQ/train_v10'); OUT=f'{B}/ckpt/r_{a.tag}'; os.makedirs(OUT,exist_ok=True)
dev=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
D={os.path.basename(p)[:-4]:dict(np.load(p)) for p in glob.glob(f'{B}/ckpt/seq/*.npz')}
QA={'own_20260529_0003','own_20260529_0004','own_20260529_0006','own_20260529_0008','own_20260906_0615','own_20260906_0620'}
SS=[k for k in D if k.startswith('sstrain__')]; AM=sorted(k for k in D if k.startswith('own__') and k[5:] not in QA)
mu=np.concatenate([D[k]['X'] for k in SS+AM]).mean(0); sd=np.concatenate([D[k]['X'] for k in SS+AM]).std(0)+1e-3
class TCN(nn.Module):
    def __init__(s,ci,ch,L=8):
        super().__init__(); s.inp=nn.Conv1d(ci,ch,1)
        s.layers=nn.ModuleList([nn.Conv1d(ch,ch,3,padding=2**i,dilation=2**i) for i in range(L)])
        s.mix=nn.ModuleList([nn.Conv1d(ch,ch,1) for _ in range(L)]); s.drop=nn.Dropout(0.2)
        s.hit=nn.Conv1d(ch,1,1); s.side=nn.Conv1d(ch,1,1)
    def forward(s,x):
        h=s.inp(x)
        for c,m in zip(s.layers,s.mix): h=h+s.drop(m(Fn.relu(c(h))))
        return s.hit(h)[:,0],s.side(h)[:,0]
def batch(keys,weights,rng,n=16,T=256):
    xs,ys,ss,ms=[],[],[],[]
    for k in rng.choices(keys,weights=weights,k=n):
        d=D[k]; L=len(d['y'])
        if L<=T: i=0
        else:
            ok=np.where(d['mask'][:L-T]>0)[0]; i=int(rng.choice(ok)) if len(ok) else rng.randrange(L-T)
        sl=slice(i,i+T); x=(d['X'][sl]-mu)/sd; pad=T-len(x)
        f=lambda v,val: np.pad(v,((0,pad),)+((0,0),)*(v.ndim-1),constant_values=val)
        xs.append(f(x,0)); ys.append(f(d['y'][sl],0)); ss.append(f(d['side'][sl].astype(np.int64),-1)); ms.append(f(d['mask'][sl],0))
    t=lambda v,dt: torch.tensor(np.stack(v),dtype=dt,device=dev)
    return t(xs,torch.float32).transpose(1,2),t(ys,torch.float32),t(ss,torch.int64),t(ms,torch.float32)
def train(model,keys,weights,steps,seed,lr=1e-3):
    rng=random.Random(seed); opt=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=1e-4); model.train()
    for it in range(steps):
        x,y,s,m=batch(keys,weights,rng); hl,sl=model(x)
        l1=(Fn.binary_cross_entropy_with_logits(hl,y,pos_weight=torch.tensor(4.0,device=dev),reduction='none')*m).sum()/m.sum().clamp(min=1)
        sm=(s>=0)&(m>0); l2=Fn.binary_cross_entropy_with_logits(sl[sm],s[sm].float()) if sm.any() else 0*l1
        loss=l1+0.5*l2; opt.zero_grad(); loss.backward(); opt.step()
    return model
@torch.no_grad()
def predict(model,k):
    model.eval(); x=torch.tensor((D[k]['X']-mu)/sd,dtype=torch.float32,device=dev).T[None]; hl,sl=model(x)
    return torch.sigmoid(hl)[0].cpu().numpy(),torch.sigmoid(sl)[0].cpu().numpy()
t0=time.time(); P={}
if a.ssoof:
    folds=[SS[i::a.ssoof] for i in range(a.ssoof)]
    for seed in range(a.seeds):
        for fi,fold in enumerate(folds):
            torch.manual_seed(seed); m=TCN(27,a.ch).to(dev); tr=[k for k in SS if k not in fold]
            train(m,tr,[1]*len(tr),a.pre,seed)
            train(m,tr+AM,[1]*len(tr)+[a.own_w*len(tr)/len(AM)/4]*len(AM),a.ft,seed+100,lr=5e-4)
            for k in fold:
                h,s=predict(m,k); P.setdefault('ssoof__'+k[9:],[]).append((h,s))
            print('fold',fi,'seed',seed,round(time.time()-t0),flush=True)
    np.savez_compressed(f'{OUT}/probs_ssoof.npz',**{f'{k}__hit':np.mean([h for h,_ in v],0) for k,v in P.items()},**{f'{k}__far':np.mean([s for _,s in v],0) for k,v in P.items()})
    sys.exit()
for seed in range(a.seeds):
    torch.manual_seed(seed); base=TCN(27,a.ch).to(dev); train(base,SS,[1]*len(SS),a.pre,seed)
    st=base.state_dict(); print('seed',seed,'pretrained',round(time.time()-t0),flush=True)
    for v in AM+['ALL']:
        m=TCN(27,a.ch).to(dev); m.load_state_dict(st); tr=[k for k in AM if k!=v]
        train(m,SS+tr,[1]*len(SS)+[a.own_w*len(SS)/max(len(tr),1)/4]*len(tr),a.ft,seed+100,lr=5e-4)
        targets=[v] if v!='ALL' else [k for k in D if not (k.startswith('own__') and k[5:] not in QA) and not k.startswith('sstrain__')]
        for k in targets:
            h,s=predict(m,k); P.setdefault(k,[]).append((h,s))
        print(' ',v,round(time.time()-t0),flush=True)
np.savez_compressed(f'{OUT}/probs.npz',**{f'{k}__hit':np.mean([h for h,_ in v],0) for k,v in P.items()},**{f'{k}__far':np.mean([s for _,s in v],0) for k,v in P.items()})
print('done',round(time.time()-t0))
