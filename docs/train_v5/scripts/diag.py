import sys; sys.path.insert(0,'/Users/mac/Desktop/RallyIQ/train_v5/scripts')
import rt
from rallyiq.perception.shuttle.tracker import *
k,t0,t1=sys.argv[1],float(sys.argv[2]),float(sys.argv[3])
c=rt.Ctx(k);fps=c.meta.fps
T=ShuttleTracker(fps,c.meta.width,c.meta.height,rt.S.tracker_config({},.2),region=c.region,support=c.support,rest=c.rest,strict_support=c.strict,held=c.held)
res=T.track(c.cands,c.meta.frame_count)
sel={p.frame:p for p in res.points}
for f in range(round(t0*fps),round(t1*fps)+1):
    cs=c.cands.get(f,[]); p=sel.get(f)
    print(f,f'{f/fps:.2f}',' | '.join(f"{x:.0f},{y:.0f} s{s:.2f} r{c.region(f,x,y):.2f}{'H' if T.hot_cell(x,y) in T.hot_cells else ''}{'N' if c.support(f,x,y) else ''}{'C' if c.strict(f,x,y) else ''}" for x,y,s in cs),' => ',f'{p.x:.0f},{p.y:.0f} {p.source[:3]}' if p else '-')
print([l for l in T.log if l[1]>=t0*fps and l[0]<=t1*fps])
