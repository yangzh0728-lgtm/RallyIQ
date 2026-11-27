"""Score fixed scorers on old (v3) and re-timed (v4) qa6 truth. Predictions are cached in ckpt/preds/<name>.json.
  rescore.py NAME [--lovo DIR] [--set k=v ...]   (DIR holds lovo_<video>.json models; otherwise --set hit_scorer=...)"""
import sys, os, json, argparse
from dataclasses import replace
from pathlib import Path
R = Path('~/Desktop/RallyIQ/train_v7/repo').expanduser(); sys.path[:0] = [str(R), str(R / 'scripts')]
import amateur_hits as AH
from rallyiq.events.config import EventsConfig
ap = argparse.ArgumentParser(); ap.add_argument('name'); ap.add_argument('--lovo'); ap.add_argument('--set', action='append', default=[])
ap.add_argument('--labels', default='~/Desktop/RallyIQ/train_v3/labels/amateur_v2.json'); ap.add_argument('--inputs'); ap.add_argument('--force', action='store_true')
a = ap.parse_args()
B = Path('~/Desktop/RallyIQ/train_v7').expanduser()
labels = Path(a.labels).expanduser()
cache = B / 'ckpt/preds' / f'{a.name}.json'
roots = {'v3truth': Path('~/Desktop/RallyIQ/train_v3/inputs').expanduser(), 'v4truth': Path(a.inputs).expanduser()}
roots.pop('v3truth')  # new videos only exist under train_v4/inputs
clips4 = AH.load_clips(roots['v4truth'], labels)
if cache.exists() and not a.force:
    preds = json.loads(cache.read_text())
else:
    preds = {}
    for c in clips4:
        over = AH.parse_set(a.set)
        if a.lovo:
            lp = Path(a.lovo).expanduser() / f'lovo_{c.cid}.json'
            # videos without a LOVO model (round-4 videos) were never trained on: the full model is held out there
            over['hit_scorer'] = str(lp if lp.exists() else Path(a.lovo).expanduser() / 'full.json')
        preds[c.cid] = c.predict_full(replace(EventsConfig(), **{**AH.AMATEUR_BASE, **over}))
    cache.write_text(json.dumps(preds))
out = {}
for k, root in roots.items():
    clips = clips4 if k == 'v4truth' else AH.load_clips(root, labels)
    md, js = AH.report(clips, preds)
    out[k] = js['total']
    print(f'## {a.name} on {k}\n{md}\n')
(B / 'logs').mkdir(exist_ok=True)
with open(B / 'logs/rescore.jsonl', 'a') as f:
    f.write(json.dumps({'name': a.name, **{k: {s: {m: round(v[s][m], 4) if isinstance(v[s][m], float) else v[s][m] for m in v[s]} for s in v} for k, v in out.items()}}) + '\n')
