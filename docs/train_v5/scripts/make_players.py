import sys; sys.path.insert(0,'/Users/mac/Desktop/RallyIQ/train_v5/scripts')
import rt,replay,os,numpy as np,offcourt
from rallyiq.contracts import PlayerTracks
from rallyiq.perception.players.geometry import apply_homography,court_side,PlayArea
keys=sys.argv[1:] or list(rt.IDS)
for k in keys:
    out=f'{rt.B}/players_v5/{k}.json'
    if os.path.exists(f'{rt.B}/dumps/{k}.npz'):
        t,r=replay.replay(k,'auto'); print(k,'replayed',r.match_format,r.format_decision,flush=True)
    elif os.path.exists(f'{rt.B}/inputs/fixcal/{k}.json'):
        cal=replay.calib(k);t=PlayerTracks.model_validate_json(rt.rd(rt.art(k),'players'));area=PlayArea()
        for fr in t.frames:
            H=np.array(cal.segment_for(fr.frame).homography)
            for p in fr.players:
                if p.foot_px:
                    c=apply_homography(H,np.array([p.foot_px]))[0]; p.foot_court=[round(float(v),3) for v in c]; p.side=court_side(c); p.on_court=area.distance_outside(c)==0.0
        print(k,'reprojected')
    else: continue
    open(out,'w').write(t.model_dump_json()); m=rt.VideoMeta.model_validate_json(rt.rd(rt.art(k),'probe')); print('  ',offcourt.stat(k,t,m.fps),flush=True)
