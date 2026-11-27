"""hmetric.py PLAYERS_DIR PREDS_NAME: hitter attribution against labels/hitter_px.json.
PLAYERS_DIR/<key>.json are the tracks the predictions were made with; PREDS in train_v5/ckpt/preds or train_v6/ckpt/preds."""
import json,os,sys,numpy as np
B5=os.path.expanduser('~/Desktop/RallyIQ/train_v5'); B6=os.path.expanduser('~/Desktop/RallyIQ/train_v6'); B7=os.path.expanduser('~/Desktop/RallyIQ/train_v7')
def iou(a,b):
    w=min(a[2],b[2])-max(a[0],b[0]);h=min(a[3],b[3])-max(a[1],b[1])
    if w<=0 or h<=0: return 0
    i=w*h; return i/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-i)
def evaluate(pdir,preds,keys=None,verbose=False):
    TR=json.load(open(f'{B6}/labels/hitter_px.json')); tot={'n':0,'tracked':0,'det_unassigned':0,'nodet':0,'matched':0,'ok':0,'wrong_tracked':0,'wrong_untracked':0}; per={}
    for k,rows in TR.items():
        if keys and k not in keys: continue
        T=json.load(open(f'{pdir}/{k}.json')); PB={f['frame']:[p for p in f['players'] if not p.get('interpolated')] for f in T['frames']}
        d=np.load(f'{B5}/dumps/{k}.npz'); fr=d['frame']; bb=d['bbox']
        c={x:0 for x in tot}
        for r in rows:
            if r['bbox'] is None: continue
            c['n']+=1; tp=None
            for e in (() if r['src'].startswith('nodet') else (0,-2,2)):
                best=max(PB.get(r['frame']+e,[]),key=lambda p:iou(p['bbox'],r['bbox']),default=None)
                if best is not None and iou(best['bbox'],r['bbox'])>.3: tp=best['player_id']; break
            if tp: c['tracked']+=1
            else:
                dets=[bb[i] for i in np.where(fr==r['frame'])[0]]
                c['det_unassigned' if any(iou(b,r['bbox'])>.3 for b in dets) else 'nodet']+=1
            m=min(preds.get(k,[]),key=lambda p:abs(p['t']-r['t']),default=None)
            if m is None or abs(m['t']-r['t'])>.2: continue
            c['matched']+=1
            if tp and m['hitter']==tp: c['ok']+=1
            elif tp: c['wrong_tracked']+=1
            else: c['wrong_untracked']+=1
            if verbose and not (tp and m['hitter']==tp): print(' ',k[-7:],r['t'],r['side'],'truth',tp,'pred',m['hitter'],r['src'][:50])
        per[k]=c; tot={x:tot[x]+c[x] for x in tot}
    return tot,per
if __name__=='__main__':
    pdir=sys.argv[1]; name=sys.argv[2]
    pp=next(p for p in (f'{B7}/ckpt/preds/{name}.json',f'{B6}/ckpt/preds/{name}.json',f'{B5}/ckpt/preds/{name}.json') if os.path.exists(p))
    tot,per=evaluate(pdir,json.load(open(pp)),verbose='-v' in sys.argv)
    for k,c in per.items(): print(k,c)
    print('TOTAL',tot,'untracked frac %.3f'%(1-tot['tracked']/tot['n']),'attr acc %.3f'%(tot['ok']/max(tot['matched'],1)))
