import json, os, sys
sys.path.insert(0, os.path.expanduser('~/Desktop/RallyIQ/train_v4/repo/scripts/train_v4'))
import sheets
B = os.path.expanduser('~/Desktop/RallyIQ/train_v4')
c = json.load(open(f'{B}/labels/r4c/candidates.json'))
for key, rows in c['cands'].items():
    items = []
    for i, h in enumerate(rows):
        nt = rows[i + 1]['t'] if i + 1 < len(rows) and rows[i + 1]['t'] - h['t'] < 3 else None
        items.append({'key': key, 't': h['t'], 'hitter': h['hitter'], 'side': h['side'], 'next_t': nt,
                      'title': f"r4c {key} #{i} t={h['t']:.2f} src={h['src']} det {h['hitter']}/{h['side']}"})
    print(sheets.render(items, f'{B}/review/r4c', key))
