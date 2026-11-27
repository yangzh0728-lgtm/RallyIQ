"""dump6.py KEY...: pose dump (v5 dump.py) for the fresh videos in inputs/d/<KEY> -> dumps/KEY.npz"""
import sys,os,time,numpy as np
sys.path.insert(0,os.path.expanduser('~/Desktop/RallyIQ/train_v6/repo'))
from rallyiq.contracts import CourtCalibration
from rallyiq.perception.players import runner as R
from rallyiq.perception.players.appearance import colour_histogram
from rallyiq.perception.players.pose import RTMOOnnx
from rallyiq.perception.players.geometry import far_court_roi
B6=os.path.expanduser('~/Desktop/RallyIQ/train_v6')
est=RTMOOnnx(os.path.expanduser('~/RallyIQ/weights/rtmo-s_body7-640x640.onnx')); cfg=R.PlayerTrackingConfig()
for k in sys.argv[1:]:
    d=f'{B6}/inputs/d/{k}'; cal=CourtCalibration.model_validate_json(open(f'{d}/calibration.json').read())
    r={q:[] for q in('frame','bbox','kps','score','hist')};t0=time.time();n=0
    for i,t,img in R.iter_video(f'{d}/source.mp4',2):
        s=cal.segment_for(i); H=np.array(s.homography) if s and s.status!='undeterminable' and s.homography else None
        roi=None
        if H is not None:
            roi=far_court_roi(H,(img.shape[1],img.shape[0]))
            if roi is not None:
                need=est.far_player_px(img.shape,roi); roi=roi if need is not None and need<cfg.far_boost_max_px else None
        dets=est(img,far_roi=roi) if roi is not None else est(img)
        n+=1
        if n%1000==0: print(k,i,round(n/(time.time()-t0),1),'fps',flush=True)
        for dd in dets:
            if dd.score<.2: continue
            h=colour_histogram(img,dd)
            r['frame'].append(i);r['bbox'].append(dd.bbox);r['kps'].append(dd.keypoints);r['score'].append(dd.score);r['hist'].append(h if h is not None else np.full(144,np.nan))
    np.savez_compressed(f'{B6}/dumps/{k}.npz',frame=np.array(r['frame'],np.int32),bbox=np.array(r['bbox'],np.float32),kps=np.array(r['kps'],np.float32),score=np.array(r['score'],np.float32),hist=np.array(r['hist'],np.float16))
    print(k,'done',n,round(time.time()-t0),'s',flush=True)
