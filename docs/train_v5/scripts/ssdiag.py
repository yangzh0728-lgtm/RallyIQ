import sys; sys.path.insert(0,'/Users/mac/Desktop/RallyIQ/train_v5/scripts')
import rt,json
idx=json.load(open(rt.H+'/train_v3/inputs/public/shuttleset_test_v3/index.json'))
for cid in sys.argv[1:]:
    ad=idx[cid]['artifacts'];vid=ad.split('/')[-2]
    c=rt.Ctx('x',ad,f'{rt.B}/inputs/ss_cands/shuttleset_test_v3/{vid}.json'); o=c.own
    class TT(rt.ShuttleTracker):
        def _ownership(s,ts):
            r=super()._ownership(ts); print(cid,c.meta.width,c.meta.fps,'prior',s.foreign_prior)
            kept={id(x) for x in r}
            for t in ts:
                if len(t.frames)<4: continue
                f=o.features(t.frames,t.xy,s._turns(t))
                if f and o.logit(t.frames,t.xy,s._turns(t))>-.5: print(t.start,t.end,len(t.frames),[tuple(map(int,q)) for q in (t.xy[0],t.xy[len(t.xy)//2],t.xy[-1])],{a:round(b,2) for a,b in f.items()},o.strip(t.frames[len(t.frames)//2]))
            return r
    T=TT(c.meta.fps,c.meta.width,c.meta.height,rt.S.tracker_config({},.2),region=c.region,support=c.support,rest=c.rest,strict_support=c.strict,held=c.held,foreign=o.logit,contact=o.contact); sx=c.meta.width/c.size[0]
    T.track({f:[(x*sx,y*sx,s) for x,y,s in v] for f,v in c.cands.items()},c.meta.frame_count)
