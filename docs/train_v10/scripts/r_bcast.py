"""r_bcast.py: ShuttleSet-only TCN (config A): 4-fold OOF on ShuttleSet train for decoding choices, and 3-seed full models
saved as torch state dicts (ckpt/r_bcast/seed<k>.pt) with the feature normalisation -> probs for sstest."""
import os,sys,glob,random,time,json
import numpy as np, torch
sys.argv=[sys.argv[0],'bc','--seeds','3']
src=open(os.path.expanduser('~/Desktop/RallyIQ/train_v10/scripts/r_train.py')).read().split('t0=time.time(); P={}')[0]
exec(src)
OUTB=f'{B}/ckpt/r_bcast'; os.makedirs(OUTB,exist_ok=True)
mu=np.concatenate([D[k]['X'] for k in SS]).mean(0); sd=np.concatenate([D[k]['X'] for k in SS]).std(0)+1e-3
globals()['mu'],globals()['sd']=mu,sd
P={}; t0=time.time(); folds=[SS[i::4] for i in range(4)]
for seed in range(3):
    for fi,fold in enumerate(folds):
        torch.manual_seed(seed); m=TCN(27,64).to(dev); tr=[k for k in SS if k not in fold]; train(m,tr,[1]*len(tr),3000,seed)
        for k in fold:
            h,s=predict(m,k); P.setdefault(k,[]).append((h,s))
    torch.manual_seed(seed); m=TCN(27,64).to(dev); train(m,SS,[1]*len(SS),3000,seed)
    torch.save(m.state_dict(),f'{OUTB}/seed{seed}.pt')
    for k in [k for k in D if k.startswith('sstest__')]:
        h,s=predict(m,k); P.setdefault(k,[]).append((h,s))
    print('seed',seed,round(time.time()-t0),flush=True)
np.savez(f'{OUTB}/norm.npz',mu=mu,sd=sd)
np.savez_compressed(f'{OUTB}/probs.npz',**{f'{k}__hit':np.mean([h for h,_ in v],0) for k,v in P.items()},**{f'{k}__far':np.mean([s for _,s in v],0) for k,v in P.items()})
print('done')
