"""athit.py INPUTS: share of labelled positive hits with no track point within +-0.1 s, per video."""
import json,sys,os
L=json.load(open('/Users/mac/Desktop/RallyIQ/train_v5/labels/amateur_v4.json'))
inp=sys.argv[1]; tot=[0,0]
for k,hs in L['hits'].items():
    S=json.load(open(f'{inp}/new_runs/{k}/shuttle.json')); fps=json.load(open(f'{inp}/new_runs/{k}/probe.json'))['fps']
    fr={p['frame'] for p in S['points']}
    pos=[h for h in hs if h['status'] in('reviewed_sure','reviewed_unsure')]
    miss=[h['t'] for h in pos if not any(round(h['t']*fps)+e in fr for e in range(-int(.1*fps),int(.1*fps)+1))]
    tot[0]+=len(pos);tot[1]+=len(miss); print(k,len(pos),len(miss),miss[:8])
print('ALL',tot,round(tot[1]/tot[0],3))
