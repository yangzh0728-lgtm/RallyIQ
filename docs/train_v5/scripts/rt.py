"""Retrack: rebuild shuttle track from saved candidates. retrack(key, **tracker overrides) -> (track, tracker)"""
import sys,os,json,gzip
H=os.path.expanduser('~/Desktop/RallyIQ'); R=os.environ.get('RTREPO',H+'/train_v5/repo')
sys.path[:0]=[R,R+'/scripts']
from pathlib import Path
from rallyiq.contracts import CourtCalibration, PlayerTracks, ShuttleTrack, VideoMeta
from rallyiq.perception.shuttle import stage as S
from rallyiq.perception.shuttle.tracker import ShuttleTracker
B=H+'/train_v5'
IDS={l.split()[0]:l.split() for l in open(B+'/ids.txt')}
def art(k): return IDS[k][-1].replace('~',os.path.expanduser('~'))
def rd(d,n):
    for p in (f'{d}/{n}.json',f'{d}/{n}.json.gz'):
        if os.path.exists(p): return gzip.open(p,'rt').read() if p.endswith('gz') else open(p).read()
class Ctx:
    def __init__(s,k,artdir=None,cands=None):
        s.k=k; d=artdir or art(k); s.meta=VideoMeta.model_validate_json(rd(d,'probe'))
        c=rd(d,'calibration'); s.cal=CourtCalibration.model_validate_json(c) if c else None
        s.pl=PlayerTracks.model_validate_json(rd(d,'players'))
        s.old=ShuttleTrack.model_validate_json(rd(d,'shuttle'))
        s.cands,s.size,s.det=S.load_candidates(Path(cands or f'{B}/inputs/cands/{k}.json'))
        s.region,s.support,s.mode=S.court_lock(s.cal,s.pl); s.rest=S.rest_area(s.cal,s.pl)
        s.strict,s.held=S.contact_tests(s.pl,s.meta,'box')
        if hasattr(S,'CourtOwnership'):
            o=S.CourtOwnership(s.pl,s.meta.fps,s.meta.width); s.own=o if o else None
    def run(s,peak=0.2,own=True,**over):
        cfg=S.tracker_config(over,peak)
        kw={'own':s.own} if own else {}
        return S.build_track(s.meta,s.cands,s.size[0],s.size[1],s.det,cfg,s.region,s.support,s.rest,s.strict,s.held,**kw)
