"""Bystander counts: player frames (in labelled windows) whose feet are past a sideline by >0.3 m, or past a baseline by >2.5 m,
and runs (pid,track) with >50% such frames. old = shipped players, new = players_v5."""
import sys,os; sys.path.insert(0,'/Users/mac/Desktop/RallyIQ/train_v5/scripts')
import rt,json,replay,numpy as np
from pathlib import Path
import amateur_hits as AH
P4=Path(rt.H+'/train_v4'); TR=AH.load_truth(P4/'inputs',P4/'labels/amateur_v3.json')
def st(k,t,cal,fps):
    W=TR[k]['windows'];fr=0;off=0;runs={}
    for f in t.frames:
        inw=any(a<=f.frame/fps<=b for a,b in W)
        H=np.array(cal.segment_for(f.frame).homography)
        for p in f.players:
            if not p.player_id or p.interpolated or p.foot_px is None: continue
            q=H@[*p.foot_px,1];x,y=q[:2]/q[2]
            o=max(-x,x-6.1)>.3 or max(-y,y-13.4)>2.5
            r=runs.setdefault((p.player_id,p.track_id),[0,0]);r[0]+=1;r[1]+=o
            if inw: fr+=1;off+=o
    return off,fr,sum(1 for a,b in runs.values() if a>=30 and b>.5*a)
tot={'old':[0,0,0],'new':[0,0,0]}
for k in rt.IDS:
    fps=float(rt.IDS[k][2]);cal=replay.calib(k);row=[k]
    for n,src in (('old',rt.rd(rt.art(k),'players')),('new',open(f'{rt.B}/players_v5/{k}.json').read() if os.path.exists(f'{rt.B}/players_v5/{k}.json') else None)):
        if src is None: row.append('-');continue
        t=rt.PlayerTracks.model_validate_json(src);s=st(k,t,cal,fps);row.append(f'{t.match_format[:1]} {s[0]}/{s[1]} runs {s[2]}')
        for i in range(3): tot[n][i]+=s[i]
    print(*row,flush=True)
print('TOTAL',tot)
