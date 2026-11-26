"""Pose dump: dump.py KEY [maxframe] -> dumps/KEY.npz (all raw detections, stride 2, torso histograms)."""
import sys; sys.path.insert(0,'/Users/mac/Desktop/RallyIQ/train_v5/scripts'); sys.path.insert(0,'/Users/mac/Desktop/RallyIQ/train_v4/scripts')
import rt,os,time,numpy as np
from rallyiq.perception.players import runner as R
from rallyiq.perception.players.appearance import colour_histogram
from rallyiq.perception.players.pose import RTMOOnnx
from rallyiq.perception.players.geometry import far_court_roi
k=sys.argv[1]; mx=int(sys.argv[2]) if len(sys.argv)>2 else None
B4=rt.H+'/train_v4/inputs'; src=next(s for s in (f'{rt.B}/inputs/sources/{k}.mp4',f'{B4}/new_runs/{k}/source.mp4',f'{B4}/sources/{k}.mp4') if os.path.exists(s))
fp=f'{rt.B}/inputs/fixcal/{k}.json'; cal=rt.CourtCalibration.model_validate_json(open(fp).read() if os.path.exists(fp) else rt.rd(rt.art(k),'calibration'))
est=RTMOOnnx(os.path.expanduser('~/RallyIQ/weights/rtmo-s_body7-640x640.onnx')); cfg=R.PlayerTrackingConfig()
r={q:[] for q in('frame','bbox','kps','score','hist')};t0=time.time();n=0
for i,t,img in R.iter_video(src,2):
    if mx and i>mx: break
    s=cal.segment_for(i); H=np.array(s.homography) if s and s.status!='undeterminable' and s.homography else None
    roi=None
    if H is not None:
        roi=far_court_roi(H,(img.shape[1],img.shape[0]))
        if roi is not None:
            need=est.far_player_px(img.shape,roi); roi=roi if need is not None and need<cfg.far_boost_max_px else None
    dets=est(img,far_roi=roi) if roi is not None else est(img)
    n+=1
    if n%500==0: print(k,i,round(n/(time.time()-t0),1),'fps',flush=True)
    for d in dets:
        if d.score<.2: continue
        h=colour_histogram(img,d)
        r['frame'].append(i);r['bbox'].append(d.bbox);r['kps'].append(d.keypoints);r['score'].append(d.score);r['hist'].append(h if h is not None else np.full(144,np.nan))
np.savez_compressed(f'{rt.B}/dumps/{k}{"_t" if mx else ""}.npz',frame=np.array(r['frame'],np.int32),bbox=np.array(r['bbox'],np.float32),kps=np.array(r['kps'],np.float32),score=np.array(r['score'],np.float32),hist=np.array(r['hist'],np.float16))
print(k,'done',n,round(time.time()-t0),'s')
