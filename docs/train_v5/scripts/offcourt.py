import sys; sys.path.insert(0,'/Users/mac/Desktop/RallyIQ/train_v5/scripts')
import rt,json,numpy as np
from collections import Counter
from pathlib import Path
import amateur_hits as AH
B4=Path(rt.H+'/train_v4'); TR=AH.load_truth(B4/'inputs',B4/'labels/amateur_v3.json')
def out(fc): 
    x,y=fc; return max(-x,x-6.1,0),max(-y,y-13.4,0)
def stat(k,pl,fps):
    W=TR[k]['windows']; n=0;off=Counter();tot=Counter();jump=Counter();last={};wf=0;woff=Counter();wtot=Counter()
    for fr in pl.frames:
        inw=any(a<=fr.frame/fps<=b for a,b in W); wf+=inw
        for p in fr.players:
            if not p.player_id or p.foot_court is None: continue
            s,e=out(p.foot_court); o=s>.6 or e>2.0
            tot[p.player_id]+=1; off[p.player_id]+=o
            if inw: wtot[p.player_id]+=1; woff[p.player_id]+=o
            q=last.get(p.player_id)
            if q and fr.frame-q[0]<=fps*1.5:
                d=np.hypot(p.foot_court[0]-q[1][0],p.foot_court[1]-q[1][1])
                if d>1.5+8*(fr.frame-q[0])/fps: jump[p.player_id]+=1
            last[p.player_id]=(fr.frame,p.foot_court)
    return dict(fmt=pl.match_format,frames=len(pl.frames),tot=dict(tot),off={i:f'{off[i]/max(tot[i],1):.1%}' for i in tot},winframes=wf,woff={i:f'{woff[i]}/{wtot[i]}' for i in sorted(wtot)},jump=dict(jump))
if __name__=='__main__':
    for k in rt.IDS:
        c=rt.PlayerTracks.model_validate_json(rt.rd(rt.art(k),'players')); m=rt.VideoMeta.model_validate_json(rt.rd(rt.art(k),'probe'))
        print(k,stat(k,c,m.fps))
