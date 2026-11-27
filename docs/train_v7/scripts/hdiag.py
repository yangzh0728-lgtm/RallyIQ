"""hdiag.py PREDS: per matched truth hit in doubles, features of the predicted hitter and their teammate ->
ckpt/hdiag_<PREDS>.json (prox at hit, min d_rel over +-dt, p_swing / wrist_v of each own series)."""
import sys,os,json,numpy as np
from pathlib import Path
R=Path('~/Desktop/RallyIQ/train_v6/repo').expanduser(); sys.path[:0]=[str(R),str(R/'scripts'),os.path.dirname(__file__)]
import amateur_hits as AH, hmetric
from rallyiq.events.config import EventsConfig
from rallyiq.events.players import PoseIndex
from rallyiq.events import pose_hits as PH
B6=Path('~/Desktop/RallyIQ/train_v6').expanduser(); B5=Path('~/Desktop/RallyIQ/train_v5').expanduser()
name=sys.argv[1]; inp=Path(sys.argv[2] if len(sys.argv)>2 else B5/'inputs_C')
preds=json.load(open(next(p for p in (B6/f'ckpt/preds/{name}.json',B5/f'ckpt/preds/{name}.json') if p.exists())))
TR=json.load(open(B6/'labels/hitter_px.json'))
cfg=EventsConfig(); model=PH.get_model('auto')
clips={c.cid:c for c in AH.load_clips(inp,B5/'labels/amateur_v4.json',only=list(TR))}
rows=[]
for k,trs in TR.items():
    c=clips[k]; P=json.load(open(inp/'new_runs'/k/'players.json')); PB={f['frame']:[p for p in f['players'] if not p.get('interpolated')] for f in P['frames']}
    team={}
    for f in P['frames']:
        for p in f['players']: team[p['player_id']]=p.get('team')
    idx=PoseIndex(c.poses,cfg); ser=PH.side_series(c.poses); pts=[p for p in c.shuttle if p.visible and p.x is not None]; ts=np.array([p.t_ms for p in pts])
    fms=1000/c.fps
    for r in trs:
        if r['bbox'] is None or r['src'].startswith('nodet'): continue
        tp=None
        for e in (0,-2,2):
            b=max(PB.get(r['frame']+e,[]),key=lambda p:hmetric.iou(p['bbox'],r['bbox']),default=None)
            if b is not None and hmetric.iou(b['bbox'],r['bbox'])>.3: tp=b['player_id'];break
        m=min(preds.get(k,[]),key=lambda p:abs(p['t']-r['t']),default=None)
        if m is None or abs(m['t']-r['t'])>.2 or tp is None or m['hitter'] is None: continue
        ph=m['hitter']; mates=[q for q,t in team.items() if t==team.get(ph) and q!=ph]
        tm=m['t']*1000; feats={}
        for q in [ph]+mates:
            d=[]
            for j in np.where(np.abs(ts-tm)<=120)[0]:
                pr=[x for x in idx.proximities(pts[j].frame,(pts[j].x,pts[j].y)) if x.player_id==q]
                if pr: d.append((abs(ts[j]-tm),pr[0].distance_rel,pr[0].score))
            s=ser.get(PH.player_series(q,c.side_at(q,round(tm/fms)) or 'near'))
            ps=pv=None
            if s is not None and s.bodies:
                tt=np.arange(tm-250,tm+251,max(fms,20.0)); X=np.asarray([PH.window_features(s,float(t),fms) for t in tt],float)
                ps=float(model.predict(X).max()); v=X[:,PH.FEATURES.index('wrist_v_max')]; pv=float(np.nanmax(v)) if np.isfinite(v).any() else None
            d.sort(); feats[q]={'d0':d[0][1] if d else None,'s0':d[0][2] if d else None,'dmin':min(x[1] for x in d) if d else None,'p_swing':ps,'wrist_v':pv}
        rows.append({'k':k,'t':r['t'],'truth':tp,'pred':ph,'mates':mates,'f':feats})
json.dump(rows,open(B6/f'ckpt/hdiag_{name}.json','w'),indent=0)
for x in rows:
    if x['truth']!=x['pred'] or '-a' in sys.argv: print(x['k'][-7:],x['t'],'truth',x['truth'],'pred',x['pred'],{q:{a:(round(b,2) if b is not None else None) for a,b in v.items()} for q,v in x['f'].items()})
print(len(rows),'rows')
