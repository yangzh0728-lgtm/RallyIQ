import sys; sys.path.insert(0,'/Users/mac/Desktop/RallyIQ/train_v5/scripts')
import rt,metric
for k in rt.IDS:
    c=rt.ShuttleTrack.model_validate_json(rt.rd(rt.art(k),'shuttle'))
    s,w=metric.score(k,{p.frame:(p.x,p.y) for p in c.points}); print(k,s,w[:12])
