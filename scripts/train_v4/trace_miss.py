"""For each missed positive with a candidate p>=0.3 (v3b LOVO), find where it is lost: not a candidate in the
run, rejected in detect_hits (conf < min / alternation / repeats), or removed after (swing veto, rally drops)."""
import sys, json
from dataclasses import replace
from pathlib import Path
B = Path('~/Desktop/RallyIQ/train_v4').expanduser(); sys.path[:0] = [str(B / 'repo'), str(B / 'repo/scripts')]
import amateur_hits as AH
from rallyiq.events import hits as H, pipeline as PL
from rallyiq.events.config import EventsConfig
misses = [l.split() for l in sys.stdin if l.strip()]
log = {}
orig_detect, orig_resolve = H.detect_hits, H.resolve_sequence
def resolve(cands, cfg):
    out = orig_resolve(cands, cfg)
    for c in cands: log.setdefault('pre', []).append((c.t_ms, round(c.confidence, 2), H._side_key(c), [(x.pose.player_id, round(x.score, 2)) for x in c.prox[:2]] if c.prox else None, c in out))
    for c in out: log.setdefault('post', []).append((c.t_ms, round(c.confidence, 2), H._side_key(c)))
    return out
H.resolve_sequence = resolve
PL.detect_hits = H.detect_hits
clips = {c.cid: c for c in AH.load_clips(B / 'inputs', B / 'labels/amateur_v3.json')}
for key in sorted({m[0] for m in misses}):
    log.clear()
    c = clips[key]
    lp = B / 'ckpt/lovo_B5' / f'lovo_{key}.json'
    cfg = replace(EventsConfig(), **{**AH.AMATEUR_BASE, 'hit_scorer': str(lp if lp.exists() else B / 'ckpt/lovo_B5/full.json')})
    final = [p['t'] for p in c.predict_full(cfg)]
    for m in [m for m in misses if m[0] == key]:
        t = float(m[1]) * 1000
        ctx = [(round((x[0] - t) / 1000, 2),) + tuple(x[1:]) for x in log.get('pre', []) if -6000 <= x[0] - t <= 1600]
        print(key, m[1], m[2])
        for c in ctx: print('     ', c)
