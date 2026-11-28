"""convert.py RAW_DIR OUT.json [KEY...]: review lines (REVIEW_PROTOCOL.md) -> amateur-format labels {note, windows, hits}.
Window = the whole video. Status: hit+sure -> reviewed_sure; hit+unsure:(stroke|hitter|side) -> reviewed_unsure;
hit+unsure:hit or unsure_hit -> unsure_hit (ignored in scoring); not_a_hit -> not_a_hit; dup / gap_none dropped;
gap_hit -> kept only if refined (refined lines come from RAW_DIR/<KEY>.refine.txt with the same format), else unsure_hit.
Two hit labels within 0.25 s: the later one is dropped (reported)."""
import sys, json, glob, os
from pathlib import Path
B = Path('~/Desktop/RallyIQ/train_v11').expanduser()
raw, out = Path(sys.argv[1]), Path(sys.argv[2]); keys = sys.argv[3:] or sorted(Path(p).stem for p in glob.glob(str(raw / '*.txt')) if '.refine' not in p)
def parse(p):
    rows = []
    for l in open(p):
        if not l.strip() or l.startswith('#'): continue
        v = [x.strip() for x in l.split('|')]
        if len(v) < 8 or not v[0].isdigit(): print('skip', p, l[:80]); continue
        rows.append({'n': int(v[0]), 'label': v[1], 't': float(v[2]) if v[2] not in ('', '-') else None, 'side': v[3], 'hitter': v[4],
                     'stroke': v[5], 'sure': v[6], 'reason': '|'.join(v[7:])})
    return rows
res = {'note': 'train_v11 labels: whole videos, every row of review/sheets checked (REVIEW_PROTOCOL.md); see labels/raw.', 'windows': {}, 'hits': {}}
for k in keys:
    items = {it['n']: it for it in json.load(open(B / f'review/items/{k}.json'))['items']}
    dur = json.load(open(B / f'review/items/{k}.json'))['duration']
    rows = parse(raw / f'{k}.txt'); RX = {x['n']: x for x in json.load(open(B / f'review/items/{k}__refine.json'))['items']} if (B / f'review/items/{k}__refine.json').exists() else {}
    ref = {(RX[r['n']]['orig_n'], round(RX[r['n']]['orig_t'], 2)): r for r in (parse(raw / f'{k}.refine.txt') if (raw / f'{k}.refine.txt').exists() else [])}
    hs = []
    for r in rows:
        lab = r['label'].lower(); it = items.get(r['n'], {})
        if lab.startswith('dup') or lab == 'gap_none': continue
        if lab in ('gap_hit', 'missed_hit'):
            q = ref.get((r['n'], round(r['t'] or 0, 2)))
            if q and q['label'] == 'hit': r = q; lab = 'hit'
            else: lab = 'unsure_hit'
        sure = r['sure'].lower()
        if lab == 'hit':
            st = 'unsure_hit' if 'unsure:hit' in sure else ('reviewed_sure' if sure == 'sure' else 'reviewed_unsure')
        elif lab in ('not_a_hit', 'unsure_hit'): st = lab
        else: print('unknown label', k, r); continue
        stroke = r['stroke'] if r['stroke'] not in ('unknown', '-', '', 'none') else None
        hs.append({'t': r['t'] if r['t'] is not None else it.get('t'), 'frame': None, 'hitter': r['hitter'] if r['hitter'] not in ('none', '-', '') else None,
                   'side': r['side'] if r['side'] in ('near', 'far') else None, 'stroke': stroke if st != 'not_a_hit' else None,
                   'model_hitter': it.get('hitter'), 'model_side': it.get('side'), 'model_stroke': it.get('stroke'), 'source': it.get('src'),
                   'row': r['n'], 'status': st, 'hitter_sure': 'unsure:hitter' not in sure, 'stroke_sure': 'unsure:stroke' not in sure and stroke is not None,
                   'review': 'claude_v11', 'note': r['reason']})
    hs.sort(key=lambda h: h['t'])
    keep = []
    for h in hs:
        if h['status'] != 'not_a_hit' and any(abs(h['t'] - g['t']) < .25 and g['status'] != 'not_a_hit' for g in keep):
            print('drop near-duplicate', k, h['row'], h['t']); continue
        keep.append(h)
    res['windows'][k] = [[0.0, round(dur, 2)]]; res['hits'][k] = keep
    from collections import Counter
    print(k, len(keep), Counter(h['status'] for h in keep))
json.dump(res, open(out, 'w'), indent=0)
