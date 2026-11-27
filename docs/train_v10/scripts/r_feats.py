"""r_feats.py: per-frame sequence features + targets for every clip -> ckpt/seq/<set>__<cid>.npz
sets: sstrain (88 ShuttleSet train), sstest (14), own (17 own videos, amateur_v6 windows/labels), fresh (3, labels/d)."""
import os,sys,json,math
import numpy as np
R=os.path.expanduser('~/Desktop/RallyIQ/train_v10/repo'); sys.path[:0]=[R,R+'/scripts']
from pathlib import Path
import learn_hits as LH, amateur_hits as AH
from multiprocessing import Pool
OUT=Path(os.path.expanduser('~/Desktop/RallyIQ/train_v10/ckpt/seq')); OUT.mkdir(parents=True,exist_ok=True)
NF=27
def feats(shuttle,poses,fps,W,H,nframes):
    st=max(1,round(fps/30)); grid=np.arange(0,nframes,st)
    sh={o.frame:o for o in shuttle if o.visible and o.x is not None}
    pf={p.frame:p for p in poses}; pk=sorted(pf)
    X=np.zeros((len(grid),NF),np.float32); last=None; since=10; prev_w={}
    for i,g in enumerate(grid):
        o=sh.get(g) or sh.get(g+1)
        if o is not None:
            if last is not None: X[i,3]=(o.x-last[0])/W*30/(fps/st)*st; X[i,4]=(o.y-last[1])/H*30/(fps/st)*st
            X[i,0]=1; X[i,1]=o.x/W; X[i,2]=o.y/H; X[i,5]=o.confidence; last=(o.x,o.y); since=0
        else:
            since=min(since+1,10)
            if last is not None: X[i,1]=last[0]/W; X[i,2]=last[1]/H
        X[i,6]=since/10
        # pose frame nearest g within 2
        j=np.searchsorted(pk,g); cand=[pk[q] for q in (j-1,j) if 0<=q<len(pk) and abs(pk[q]-g)<=2]
        if not cand: continue
        P=pf[min(cand,key=lambda q:abs(q-g))]
        sx,sy=(last if last is not None else (W/2,H/2))
        for si,side in enumerate(('near','far')):
            best=None
            for p in P.players:
                if p.side!=side: continue
                kp=[(k.x,k.y) if k.confidence>=0.3 else None for k in p.keypoints]
                ys=[k[1] for k in kp if k]; 
                if len(ys)<4: continue
                h=max(max(ys)-min(ys),20.0)
                wr=[kp[q] for q in (9,10)]
                d=min([math.hypot(w[0]-sx,w[1]-sy) for w in wr if w] or [1e9])/h
                if best is None or d<best[0]: best=(d,p,kp,h,wr)
            if best is None: continue
            d,p,kp,h,wr=best; b=7+si*10; X[i,b]=1
            for q,w in enumerate(wr):
                if w: X[i,b+1+2*q]=np.clip((w[0]-sx)/h,-5,5); X[i,b+2+2*q]=np.clip((w[1]-sy)/h,-5,5)
            hp=[kp[q] for q in (11,12) if kp[q]]
            if hp: X[i,b+5]=np.clip((np.mean([q[0] for q in hp])-sx)/h,-5,5); X[i,b+6]=np.clip((np.mean([q[1] for q in hp])-sy)/h,-5,5)
            X[i,b+7]=h/H
            wv=[w for w in wr if w]; key=(side,p.player_id)
            if wv and key in prev_w: X[i,b+8]=np.clip(max(math.hypot(w[0]-prev_w[key][0],w[1]-prev_w[key][1]) for w in wv)/h*(fps/st)/10,0,3)
            if wv: prev_w[key]=wv[0]
            shs=[kp[q] for q in (5,6) if kp[q]]
            if shs and wv: X[i,b+9]=np.clip((min(q[1] for q in shs)-min(w[1] for w in wv))/h,-2,2)
    return grid,X
def save(name,grid,X,fps,hits,mask,extra):
    y=np.zeros(len(grid),np.float32); side=np.full(len(grid),-1,np.int8)
    st=grid[1]-grid[0] if len(grid)>1 else 1
    for f,s in hits:
        i=int(round(f/st))
        for d in (-1,0,1):
            if 0<=i+d<len(grid): y[i+d]=1; side[i+d]={'near':0,'far':1}.get(s,-1)
    np.savez_compressed(OUT/f'{name}.npz',grid=grid,X=X,y=y,side=side,mask=mask.astype(np.float32),fps=fps,**extra)
def ss(arg):
    split,c=arg; m=json.loads((c.art/'probe.json').read_text())
    grid,X=feats(c.shuttle,c.poses,c.fps,m['width'],m['height'],m['frame_count'])
    hits=[((h['t_ms']-c.offset)*c.fps/1000,h.get('side')) for h in c.label['hits']]
    save(f'{split}__{c.cid}',grid,X,c.fps,hits,np.ones(len(grid)),{'offset':c.offset}); return c.cid
B6=Path('/Users/mac/Desktop/RallyIQ/train_v6')
def fresh_truth(k):
    L=[l.split('|') for l in open(B6/f'labels/d/{k}.txt') if l[0].isdigit()]
    return {'set':'fresh','windows':json.load(open(B6/f'ckpt/d/v6/{k}.json'))['windows'],
            'labels':[{'t':float(x[2]),'kind':{'hit':'pos','not_a_hit':'neg','unsure_hit':'ign'}[x[1].strip()],'side':x[3].strip()} for x in L]}
def own(arg):
    split,k,art,truth=arg; c=AH.AmClip(k,art,truth); m=json.loads(AH.read_json(art,'probe'))
    grid,X=feats(c.shuttle,c.poses,c.fps,m['width'],m['height'],m['frame_count'])
    t=grid/c.fps; mask=np.zeros(len(grid))
    for a,b in truth['windows']: mask[(t>=a)&(t<=b)]=1
    for g in truth['labels']:
        if g['kind']=='ign': mask[np.abs(t-g['t'])<=0.2]=0
    hits=[(g['t']*c.fps,g.get('side')) for g in truth['labels'] if g['kind']=='pos']
    save(f'{split}__{k}',grid,X,c.fps,hits,mask,{'offset':0}); return k
if __name__=='__main__':
    jobs_ss=[('sstrain',c) for c in LH.load_clips(Path('/Users/mac/Desktop/RallyIQ/train_v5/inputs/pub_v5/shuttleset_train_v3'),'index.json')]
    jobs_ss+=[('sstest',c) for c in LH.load_clips(Path('/Users/mac/Desktop/RallyIQ/train_v5/inputs/pub_v5/shuttleset_test_v3'),'results.json')]
    with Pool(6) as p: print(len(p.map(ss,jobs_ss)),'ss',flush=True)
    INP=Path('/Users/mac/Desktop/RallyIQ/train_v5/inputs_C'); TR=AH.load_truth(INP,Path(os.path.expanduser('~/Desktop/RallyIQ/train_v10/labels/amateur_v6.json')))
    jobs=[('own',k,AH.art_dir(INP,k),TR[k]) for k in TR]+[('fresh',k,B6/f'inputs/d_run/new_runs/{k}',fresh_truth(k)) for k in ('own_20260529_0005','own_20260529_0007','own_20260529_0009')]
    with Pool(6) as p: print(p.map(own,jobs),flush=True)
