"""mkpass2.py KEY...: random 10% of each video's review rows (seed 11) -> review/items/KEY__pass2.json for an independent re-review."""
import sys, json, random
from pathlib import Path
B = Path('~/Desktop/RallyIQ/train_v11').expanduser()
for k in sys.argv[1:]:
    D = json.load(open(B / f'review/items/{k}.json')); it = D['items']
    rng = random.Random(f'11-{k}'); s = sorted(rng.sample(range(len(it)), max(1, round(len(it) * .1))))
    json.dump({**D, 'items': [it[i] for i in s]}, open(B / f'review/items/{k}__pass2.json', 'w'), indent=0); print(k, len(s))
