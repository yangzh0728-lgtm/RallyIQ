"""Time the events stage (segment_events + veto) on one or more videos: time_events.py KEY [KEY ...] [--profile] [--set k=v]"""
import sys, json, time, cProfile, pstats
from dataclasses import replace
from pathlib import Path
B = Path('~/Desktop/RallyIQ/train_v4').expanduser(); sys.path[:0] = [str(B / 'repo'), str(B / 'repo/scripts')]
import amateur_hits as AH
from rallyiq.contracts import PlayerTracks, ShuttleTrack, CourtCalibration
from rallyiq.events.adapters import calibration_segments, poses_from_contract, shuttle_from_contract
from rallyiq.events.config import EventsConfig
from rallyiq.events.pipeline import off_court_views, segment_events
args = [a for a in sys.argv[1:] if not a.startswith('--') and '=' not in a]
sets = [a for a in sys.argv[1:] if '=' in a]
cfg = replace(EventsConfig(), **{**AH.AMATEUR_BASE, **AH.parse_set(sets)})
tot = 0; nh = 0
for key in args:
    d = AH.art_dir(B / 'inputs', key); probe = json.loads(AH.read_json(d, 'probe'))
    sh = ShuttleTrack.model_validate_json(AH.read_json(d, 'shuttle')); pl = PlayerTracks.model_validate_json(AH.read_json(d, 'players'))
    cj = AH.read_json(d, 'calibration'); cal = CourtCalibration.model_validate_json(cj) if cj else None
    off = off_court_views(calibration_segments(cal), cfg.court_view_min_share)
    s_in, p_in = shuttle_from_contract(sh), poses_from_contract(pl)
    t0 = time.perf_counter()
    if '--profile' in sys.argv:
        pr = cProfile.Profile(); pr.enable()
    r = segment_events(s_in, p_in, cfg, video_end_ms=float(probe['duration_ms']), off_court=off)
    dt = time.perf_counter() - t0; tot += dt; h = sum(len(x.hits) for x in r); nh += h
    if '--profile' in sys.argv:
        pr.disable(); pstats.Stats(pr).sort_stats('cumulative').print_stats(25)
    print(f'{key}: {dt:.1f}s, {len(r)} rallies, {h} hits, duration {probe["duration_ms"] / 60000:.1f} min', flush=True)
print(f'total {tot:.1f}s, {nh} hits')
