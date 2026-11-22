"""amateur_v2.json (leg 1) + stroke review (labels/strokes1_decisions.txt) + round 4 (labels/r4) -> labels/amateur_v3.json"""
import json, os, copy
B = os.path.expanduser('~/Desktop/RallyIQ/train_v4')
d = json.load(open(os.path.expanduser('~/Desktop/RallyIQ/train_v3/labels/amateur_v2.json')))
items = {it['n']: it for it in json.load(open(f'{B}/labels/strokes1_items.json'))}
n_changed = n_sure = 0
for line in open(f'{B}/labels/strokes1_decisions.txt'):
    if not line.startswith('S'): continue
    sid, st, sure, note = [x.strip() for x in line.split('|', 3)]
    it = items[int(sid[1:])]
    h = next(h for h in d['hits'][it['key']] if abs(h['t'] - it['t']) < 1e-6)
    old = h.get('stroke')
    if st == '-':
        if old is not None and h.get('stroke_sure') is not True:
            h['stroke_v3'] = old
        h['stroke'] = old if st == '-' and False else None
    else:
        h['stroke'] = st
    h['stroke_sure'] = sure == 'true'
    h['stroke_review_v4'] = note
    if 'partner' in note and 'hitter = partner' in note: h['hitter'] = None
    n_changed += old != h['stroke']; n_sure += h['stroke_sure']
n4 = 0
for rnd in ('r4', 'r4b', 'r4c'):
  r4 = [l for l in open(f'{B}/labels/{rnd}/decisions.txt') if not l.startswith('#') and l.strip()]
  cand = json.load(open(f'{B}/labels/{rnd}/candidates.json'))
  for l in r4:
      key, idx, t, status, hitter, side, stroke, ss, note = [x.strip() for x in l.split('|', 8)]
      c = cand['cands'][key][int(idx)]
      assert abs(c['t'] - float(t)) < 0.011, (key, idx, c['t'], t)
      if rnd == 'r4' and '14' == idx and key == 'own_20260906_0621': hitter, side = 'P3', 'near'
      st = {'hit': 'reviewed_sure' if ss == 'true' else 'reviewed_unsure', 'not_a_hit': 'not_a_hit', 'unsure_hit': 'unsure_hit'}[status]
      d['windows'].setdefault(key, [])
      for w in cand['windows'][key]:
        if list(w) not in d['windows'][key]: d['windows'][key].append(list(w))
      d['hits'].setdefault(key, []).append({'t': c['t'], 'frame': None, 'hitter': None if hitter == 'null' else hitter, 'side': side,
          'stroke': None if stroke == '-' else stroke, 'stroke_sure': ss == 'true', 'hitter_sure': status == 'hit',
          'model_hitter': c['hitter'], 'model_side': c['side'], 'source': c['src'], 'status': st, 'review': 'claude_v4_' + rnd, 'note': note})
      n4 += 1
d['note'] = d.get('note', '') + ' | v4: stroke review of leg-1 hits (strokes1_decisions.txt) + round 4 (4 new 0906 videos, labels/r4)'
json.dump(d, open(f'{B}/labels/amateur_v3.json', 'w'), indent=1)
allh = [h for v in d['hits'].values() for h in v]
pos = [h for h in allh if h['status'] in ('reviewed_sure', 'reviewed_unsure')]
print('stroke changed', n_changed, 'sure in pass', n_sure, '| r4 rows', n4)
print('videos', len(d['hits']), 'positives', len(pos), 'trusted strokes', sum(1 for h in pos if h.get('stroke') and h.get('stroke_sure')),
      'typed', sum(1 for h in pos if h.get('stroke')), 'negatives', sum(1 for h in allh if h['status'] == 'not_a_hit'))
from collections import Counter
print(Counter(h['stroke'] for h in pos if h.get('stroke') and h.get('stroke_sure')))
