import sys; sys.path.insert(0,'/Users/mac/Desktop/RallyIQ/train_v5/scripts')
import rt,replay,offcourt
for a in sys.argv[1:]:
    k,f=a.split(':'); t,r=replay.replay(k,f); print(k,f,r.match_format,r.format_decision,offcourt.stat(k,t,rt.VideoMeta.model_validate_json(rt.rd(rt.art(k),'probe')).fps))
