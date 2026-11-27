"""s_train.py TAG: BST-style (small temporal CNN + attention pooling) stroke classifier on 46-frame windows. Pretrain on
ShuttleSet train strokes; per am7 video fine-tune on the other own rows (sure w=1, unsure w=0.5) and predict it (LOVO); a model
on all own rows predicts fresh and ShuttleSet test. 5 seeds averaged. -> ckpt/s_<TAG>/preds.json"""
import os,sys,json,random,time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn
tag=sys.argv[1]; SEEDS=int(os.environ.get('SEEDS',5)); B=os.path.expanduser('~/Desktop/RallyIQ/train_v10'); OUT=f'{B}/ckpt/s_{tag}'; os.makedirs(OUT,exist_ok=True)
dev=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
d=np.load(f'{B}/ckpt/s_data.npz'); W,y,g,key,sure=d['W'],d['y'],d['g'],d['key'],d['sure']
mu=W[g=='ss_train'].reshape(-1,28).mean(0); sd=W[g=='ss_train'].reshape(-1,28).std(0)+1e-3; Wn=(W-mu)/sd
K=12
class Net(nn.Module):
    def __init__(s,ch=64):
        super().__init__()
        s.c=nn.Sequential(nn.Conv1d(28,ch,3,padding=1),nn.ReLU(),nn.Conv1d(ch,ch,3,padding=2,dilation=2),nn.ReLU(),nn.Conv1d(ch,ch,3,padding=4,dilation=4),nn.ReLU())
        s.att=nn.Conv1d(ch,1,1); s.drop=nn.Dropout(0.3); s.out=nn.Linear(ch,K)
    def forward(s,x):
        h=s.c(x); a=torch.softmax(s.att(h),-1); return s.out(s.drop((h*a).sum(-1)))
def fit(m,idx,wts,steps,lr,seed):
    rng=np.random.default_rng(seed); opt=torch.optim.Adam(m.parameters(),lr=lr,weight_decay=1e-3); m.train()
    p=wts/wts.sum()
    for _ in range(steps):
        b=rng.choice(idx,size=64,p=p); x=torch.tensor(Wn[b],dtype=torch.float32,device=dev).transpose(1,2)
        x=x+0.05*torch.randn_like(x)
        loss=Fn.cross_entropy(m(x),torch.tensor(y[b],device=dev)); opt.zero_grad(); loss.backward(); opt.step()
    return m
@torch.no_grad()
def prob(m,idx):
    m.eval(); return torch.softmax(m(torch.tensor(Wn[idx],dtype=torch.float32,device=dev).transpose(1,2)),-1).cpu().numpy()
SS=np.where(g=='ss_train')[0]; OWN=np.where(g=='own')[0]; vids=sorted(set(key[OWN]))
P={}; t0=time.time()
for seed in range(SEEDS):
    torch.manual_seed(seed); base=Net().to(dev); fit(base,SS,np.ones(len(SS)),3000,1e-3,seed); st=base.state_dict()
    for v in vids+['ALL']:
        tr=OWN[key[OWN]!=v] if v!='ALL' else OWN
        idx=np.concatenate([SS,tr]); w=np.concatenate([np.full(len(SS),0.5*len(tr)/len(SS)),np.where(sure[tr],1.0,0.5)])
        m=Net().to(dev); m.load_state_dict(st); fit(m,idx,w,800,3e-4,seed+50)
        te=OWN[key[OWN]==v] if v!='ALL' else np.where((g=='fresh')|(g=='ss_test'))[0]
        pr=prob(m,te)
        for i,q in zip(te,pr): P.setdefault(int(i),[]).append(q)
    # ShuttleSet-only model on ss_test (broadcast reference)
    pr=prob(base,np.where(g=='ss_test')[0])
    for i,q in zip(np.where(g=='ss_test')[0],pr): P.setdefault(('ssonly',int(i)),[]).append(q)
    print('seed',seed,round(time.time()-t0),flush=True)
json.dump({str(k):np.mean(v,0).tolist() for k,v in P.items()},open(f'{OUT}/preds.json','w'))
