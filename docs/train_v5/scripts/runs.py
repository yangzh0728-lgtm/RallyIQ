import sys; sys.path.insert(0,'/Users/mac/Desktop/RallyIQ/train_v5/scripts')
import rt,numpy as np
from rallyiq.perception.players.filtering import is_seated
from rallyiq.perception.players.models import PersonDetection
for k in sys.argv[1:]:
    t=rt.PlayerTracks.model_validate_json(open(f'{rt.B}/players_v5/{k}.json').read());fps=float(rt.IDS[k][2]);R={}
    for f in t.frames:
        for p in f.players:
            if p.player_id and p.foot_court and not p.interpolated: R.setdefault((p.player_id,p.track_id),[]).append((f.frame,p))
    print(k)
    for key,v in sorted(R.items(),key=lambda x:x[1][0][0]):
        if len(v)<45: continue
        xy=np.array([p.foot_court for _,p in v]);x=xy[:,0];o=np.maximum(np.maximum(-x,x-6.1),0)
        # windows of 3 s
        ba=np.median([(p.bbox[2]-p.bbox[0])/(p.bbox[3]-p.bbox[1]) for _,p in v])
        seat=np.mean([is_seated(PersonDetection(bbox=np.array(p.bbox),keypoints=np.array(p.keypoints),score=.9)) for _,p in v[::5]])
        ext=np.percentile(xy,95,0)-np.percentile(xy,5,0)
        w=int(3*fps/2); st=[np.ptp(xy[i:i+w],axis=0).max() for i in range(0,max(1,len(xy)-w),w)]
        if (o>0).mean()>.3 or seat>.3: print(' ',key,f'{v[0][0]/fps:.0f}-{v[-1][0]/fps:.0f}s n{len(v)} xmed {np.median(x):.2f} y {np.median(xy[:,1]):.1f} outfrac {(o>0).mean():.2f} ext {ext.round(1)} seated {seat:.2f} aspect {ba:.2f} 3s-extent {np.median(st):.2f}')
        
