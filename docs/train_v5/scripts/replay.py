"""replay.py KEY FORMAT -> PlayerTracks from a pose dump with the current tracking code. replay(key, fmt, **tracker overrides)"""
import sys; sys.path.insert(0,'/Users/mac/Desktop/RallyIQ/train_v5/scripts')
import rt,os,numpy as np
from rallyiq.perception.players import runner as R
from rallyiq.perception.players import stage as PS
from rallyiq.perception.players.models import PersonDetection
from rallyiq.perception.target_court import SegmentRegions
def calib(k):
    fp=f'{rt.B}/inputs/fixcal/{k}.json'; return rt.CourtCalibration.model_validate_json(open(fp).read() if os.path.exists(fp) else rt.rd(rt.art(k),'calibration'))
def load(k,suffix=''):
    d=np.load(f'{rt.B}/dumps/{k}{suffix}.npz'); by={}; fr=d['frame'];bb=d['bbox'];kp=d['kps'];sc=d['score'];hi=d['hist'].astype(np.float32)
    for i in range(len(fr)): by.setdefault(int(fr[i]),[]).append((PersonDetection(bbox=bb[i].copy(),score=float(sc[i]),keypoints=kp[i].copy()),None if np.isnan(hi[i][0]) else hi[i]))
    return by
def replay(k,fmt='auto',by=None,cal=None,suffix='',min_inside=.25,mod=None,**ov):
    by=by or load(k,suffix); cal=cal or calib(k); meta=rt.VideoMeta.model_validate_json(rt.rd(rt.art(k),'probe'))
    cfg=R.PlayerTrackingConfig(stride=2,match_format=fmt,min_inside_frac=min_inside); cfg.tracker.fps=meta.fps
    for a,b in ov.items(): setattr(cfg.tracker if hasattr(cfg.tracker,a) else cfg,a,b)
    cur={}
    class E:
        def __call__(s,img,far_roi=None): 
            rows=by.get(cur['f'],[]); cur['h']={id(d):h for d,h in rows}; return [d for d,_ in rows]
    R.colour_histogram=lambda img,d: cur['h'].get(id(d))
    img=np.zeros((meta.height,meta.width,3),np.uint8)
    def frames():
        for f in range(0,meta.frame_count,2):
            cur['f']=f; yield f,float(meta.t_ms(f)),img
    def H(f):
        s=cal.segment_for(f); return np.array(s.homography,float) if s is not None and s.status!='undeterminable' and s.homography else None
    reg=SegmentRegions(cal)
    cfg.far_boost='off'
    res=R.track_players(frames(),E(),H,cfg,reg.for_frame if reg else None)
    return PS.build_tracks(meta.video_id,res,'rtmo-onnx:rtmo-s_body7-640x640.onnx'),res
