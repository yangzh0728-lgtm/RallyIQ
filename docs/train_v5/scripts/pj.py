import sys; sys.path.insert(0,'/Users/mac/Desktop/RallyIQ/train_v5/scripts')
import rt,replay,offcourt
for k in sys.argv[1:]:
    by=replay.load(k)
    for ov in (dict(max_jump_m=99,side_out_s=99),dict(max_jump_m=99),dict(max_jump_m=3),dict(max_jump_m=5),dict(max_jump_m=7)):
        t,r=replay.replay(k,'auto',by=by,**ov); s=offcourt.stat(k,t,rt.VideoMeta.model_validate_json(rt.rd(rt.art(k),'probe')).fps); print(k[-4:],ov,s['tot'],s['woff'],s['jump'])
