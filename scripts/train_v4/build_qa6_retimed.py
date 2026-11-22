"""hits_truth_v2.json + labels/qa6_retime_readings.txt -> labels/qa6_retimed.json (same format).
Timed readings move t (align_src 'retimed_v4', t_unrefined keeps the old time); 'side=X' corrects the side;
'none' sets in_play None (ignored in scoring)."""
import json, re, os
B = os.path.expanduser('~/Desktop/RallyIQ/train_v4')
d = json.load(open(os.path.expanduser('~/Desktop/RallyIQ/train_v3/inputs/hits_truth_v2.json')))
rd = {}
for line in open(f'{B}/labels/qa6_retime_readings.txt'):
    if not line[0].isdigit(): continue
    i, off, conf, note = [x.strip() for x in line.split('|', 3)]
    rd[int(i)] = (off, conf, note)
n = {'timed': 0, 'none': 0, 'side': 0}
for h in d['hits']:
    if h['id'] not in rd: continue
    off, conf, note = rd[h['id']]
    h['retime_v4'] = {'reading': off, 'conf': conf, 'note': note}
    if off.startswith('none'):
        h['in_play'] = None; h['in_play_why'] = 'retime v4: no contact of the labelled side within -0.8..+0.7 s (duplicate/dead ball)'; n['none'] += 1
        continue
    m = re.match(r'([+-]?\d+\.\d+)(?:\s+side=(\w+))?', off)
    h['t_unrefined'] = h['t']; h['t'] = round(h['t'] + float(m.group(1)), 3); h['align_src'] = 'retimed_v4'; n['timed'] += 1
    if m.group(2): h['side_unrefined'] = h['side']; h['side'] = m.group(2); n['side'] += 1
d['note'] = d.get('note', '') + ' | v4 2026-11-22: the 58 unrefined sure in-play hits re-timed from frame sheets (train_v4/labels/qa6_retime_readings.txt)'
json.dump(d, open(f'{B}/labels/qa6_retimed.json', 'w'), indent=1)
print(n, len(rd))
