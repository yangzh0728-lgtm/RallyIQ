"""Score cached predictions on a subset of videos: subset_score.py LABELS KEYSPEC name1 name2 ...
KEYSPEC: 'r4' (the 4 new 0906 videos), 'old' (the 13 leg-1 videos) or comma list."""
import sys, json
from pathlib import Path
B = Path('~/Desktop/RallyIQ/train_v4').expanduser(); sys.path[:0] = [str(B / 'repo'), str(B / 'repo/scripts')]
import amateur_hits as AH
R4 = ['own_20260906_0617', 'own_20260906_0618', 'own_20260906_0619', 'own_20260906_0621']
labels, spec, names = Path(sys.argv[1]).expanduser(), sys.argv[2], sys.argv[3:]
clips = AH.load_clips(B / 'inputs', labels)
keys = R4 if spec == 'r4' else ([c.cid for c in clips if c.cid not in R4] if spec == 'old' else spec.split(','))
clips = [c for c in clips if c.cid in keys]
for n in names:
    preds = json.load(open(B / f'ckpt/preds/{n}.json'))
    s = AH.summarize({c.cid: AH.score_clip(c, preds[c.cid]) for c in clips})
    print(f"{n:28s} F1 {s['f1']:.3f} P {s['precision']:.3f} R {s['recall']:.3f} pos {s['pos']} tp {s['tp']} fp {s['fp']} dead {s['dead_fp']} side {s['side_ok']}/{s['side_n']} hitter {s['hitter_ok']}/{s['hitter_n']}")
