"""Stroke-review sheets for leg-1 hits without a trusted stroke (amateur_v2: reviewed_* and stroke_sure False).
Row title shows the current label. -> review/strokes1/<key>_NN.jpg, index in labels/strokes1_items.json"""
import json, os, sys
sys.path.insert(0, os.path.expanduser('~/Desktop/RallyIQ/train_v4/repo/scripts/train_v4'))
import sheets
B = os.path.expanduser('~/Desktop/RallyIQ/train_v4')
d = json.load(open(os.path.expanduser('~/Desktop/RallyIQ/train_v3/labels/amateur_v2.json')))
allitems = []
for key, hs in d['hits'].items():
    pos = sorted([h for h in hs if h['status'] in ('reviewed_sure', 'reviewed_unsure')], key=lambda h: h['t'])
    items = []
    for i, h in enumerate(pos):
        if h.get('stroke_sure'): continue
        nt = pos[i + 1]['t'] if i + 1 < len(pos) and pos[i + 1]['t'] - h['t'] < 3 else None
        n = len(allitems)
        it = {'n': n, 'key': key, 't': h['t'], 'hitter': h.get('hitter'), 'side': h.get('side'), 'next_t': nt, 'cur': h.get('stroke'),
              'title': f"S{n} {key} t={h['t']:.2f} {h.get('hitter')}/{h.get('side')} label={h.get('stroke')} next={'%+.2f' % (nt - h['t']) if nt else '-'}"}
        items.append(it); allitems.append(it)
    if items: print(key, len(items), sheets.render(items, f'{B}/review/strokes1', key, per=4)[-1])
json.dump(allitems, open(f'{B}/labels/strokes1_items.json', 'w'), indent=0)
