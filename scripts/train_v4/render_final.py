"""Review sheets with my final label written on, per video, for sam to spot errors from screenshots:
round 4/4b/4c decisions (every reviewed detection) and the leg-1 hits whose stroke I re-labelled.
-> review/final/<video>_NN.jpg (4 rows per sheet). Title: id | my label (hit/NOT A HIT/unsure, hitter, side, stroke,
sure?) | was (detector or leg-1 label) | my one-line reason (shortened)."""
import json, os, sys
sys.path.insert(0, os.path.expanduser('~/Desktop/RallyIQ/train_v4/repo/scripts/train_v4'))
import sheets
B = os.path.expanduser('~/Desktop/RallyIQ/train_v4')
items = {}
for rnd in ('r4', 'r4b', 'r4c'):
    cand = json.load(open(f'{B}/labels/{rnd}/candidates.json'))
    for l in open(f'{B}/labels/{rnd}/decisions.txt'):
        if l.startswith('#') or not l.strip(): continue
        key, idx, t, status, hitter, side, stroke, ss, note = [x.strip() for x in l.split('|', 8)]
        c = cand['cands'][key][int(idx)]
        lab = {'hit': 'HIT', 'not_a_hit': 'NOT A HIT', 'unsure_hit': 'UNSURE'}[status]
        if status == 'hit':
            lab += f" {hitter}/{side} {stroke if stroke != '-' else 'stroke ?'}{' (sure)' if ss == 'true' else ''}"
        items.setdefault(key, []).append({'key': key, 't': c['t'], 'hitter': hitter if hitter != 'null' else None, 'side': side,
            'title': f"{rnd}#{idx} t={c['t']:.2f} MY LABEL: {lab} | det {c['hitter']}/{c['side']} | {note[:70]}"})
s1 = {it['n']: it for it in json.load(open(f'{B}/labels/strokes1_items.json'))}
for l in open(f'{B}/labels/strokes1_decisions.txt'):
    if not l.startswith('S'): continue
    sid, st, sure, note = [x.strip() for x in l.split('|', 3)]
    it = s1[int(sid[1:])]
    items.setdefault(it['key'], []).append({'key': it['key'], 't': it['t'], 'hitter': it['hitter'], 'side': it['side'], 'next_t': it['next_t'],
        'title': f"{sid} t={it['t']:.2f} MY STROKE: {st if st != '-' else '?'}{' (sure)' if sure == 'true' else ''} | was {it['cur']} | {note[:70]}"})
out = f'{B}/review/final'
for key, its in items.items():
    its.sort(key=lambda x: x['t'])
    for i, it in enumerate(its):
        if 'next_t' not in it or it['next_t'] is None:
            nxt = [x['t'] for x in its[i + 1:] if 0.25 < x['t'] - it['t'] < 3]
            it['next_t'] = nxt[0] if nxt else None
    print(key, len(its), len(sheets.render(its, out, key, per=4)), 'sheets', flush=True)
