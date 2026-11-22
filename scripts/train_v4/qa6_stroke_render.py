"""Stroke sheets for qa6 sure in-play hits (re-timed truth) of 0906_0615 / 0906_0620 / 0529_0006. -> review/strokes_qa6/"""
import json, os, sys
sys.path.insert(0, os.path.expanduser('~/Desktop/RallyIQ/train_v4/repo/scripts/train_v4'))
import sheets
B = os.path.expanduser('~/Desktop/RallyIQ/train_v4')
v = json.load(open(f'{B}/labels/qa6_retimed.json'))['hits']
out = []
for clip in ('own_20260906_0615', 'own_20260906_0620', 'own_20260529_0006'):
    hs = sorted([h for h in v if h['clip'] == clip and h['sure'] and h['in_play'] is True], key=lambda h: h['t'])
    items = []
    for i, h in enumerate(hs):
        nt = hs[i + 1]['t'] if i + 1 < len(hs) and hs[i + 1]['t'] - h['t'] < 3 else None
        it = {'n': h['id'], 'key': clip, 't': h['t'], 'hitter': None, 'side': h['side'], 'next_t': nt,
              'title': f"Q{h['id']} {clip} t={h['t']:.2f} side={h['side']} next={'%+.2f' % (nt - h['t']) if nt else '-'}"}
        items.append(it); out.append(it)
    print(clip, len(items), sheets.render(items, f'{B}/review/strokes_qa6', clip, per=4)[-1], flush=True)
json.dump(out, open(f'{B}/labels/strokes_qa6_items.json', 'w'), indent=0)
