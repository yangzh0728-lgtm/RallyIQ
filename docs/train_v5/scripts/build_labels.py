"""amateur_v3.json + v5_decisions -> labels/amateur_v4.json (old player ids) and labels/amateur_v4_newids.json / amateur_v3_newids.json
(hitter ids of the replayed players in players_v5)."""
import sys; sys.path.insert(0,'/Users/mac/Desktop/RallyIQ/train_v5/scripts')
import rt,json,copy,os
from collections import Counter
src=json.load(open(rt.H+'/train_v4/labels/amateur_v3.json')); v4=copy.deepcopy(src); n=Counter()
def singles_hitter(pl,t,side):
    f=round(t*pl['fps'])
    for d in range(0,8):
        for g in (f-d,f+d):
            ps=[p for p in pl['by'].get(g,[]) if p.side==side and p.player_id]
            if len(ps)==1: return ps[0].player_id
    return None
def load(k,new):
    p=f'{rt.B}/players_v5/{k}.json'
    t=rt.PlayerTracks.model_validate_json(open(p).read() if new and os.path.exists(p) else rt.rd(rt.art(k),'players'))
    return dict(fps=float(rt.IDS[k][2]),by={f.frame:f.players for f in t.frames},fmt=t.match_format)
OLD={k:load(k,0) for k in src['hits']}
for l in open(rt.B+'/labels/v5_decisions.txt'):
    if l[0]=='#' or not l.strip(): continue
    k,act,t,x,side,st,note=[q.strip() for q in l.split('|',6)]; t=float(t); hs=v4['hits'][k]; n[act+':'+(x if act in('add','status') else '')]+=1
    stroke=None if st=='-' else st
    if act=='add':
        hs.append(dict(t=t,frame=None,hitter=singles_hitter(OLD[k],t,side) if x=='hit' else None,side=side,stroke=stroke,stroke_sure=False,hitter_sure=False,status={'hit':'reviewed_unsure','not_a_hit':'not_a_hit','unsure_hit':'unsure_hit'}[x],review='claude_v5',note=note,source='v5')); continue
    h=min(hs,key=lambda h:abs(h['t']-t)); assert abs(h['t']-t)<.02,(k,t,h['t'])
    if act=='drop': hs.remove(h)
    elif act=='retime': h['t_v3']=h['t'];h['t']=float(x);h['note_v5']=note
    elif act=='status':
        h['status']={'hit':'reviewed_unsure','not_a_hit':'not_a_hit','unsure_hit':'unsure_hit'}[x];h['note_v5']=note
        if x=='hit': h.update(side=side,stroke=stroke,stroke_sure=False,hitter=singles_hitter(OLD[k],t,side))
for k in v4['hits']: v4['hits'][k].sort(key=lambda h:h['t'])
v4['note']+=' | v5: hits exposed by the cleaned tracks added, 7 retimed, round 2 after the doubles swing fix, see labels/v5_decisions.txt'
json.dump(v4,open(rt.B+'/labels/amateur_v4.json','w'),indent=1); print(n)
def iou(a,b):
    w=min(a[2],b[2])-max(a[0],b[0]);h=min(a[3],b[3])-max(a[1],b[1])
    if w<=0 or h<=0: return 0
    i=w*h; return i/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-i)
def remap(d):
    d=copy.deepcopy(d); c=Counter()
    for k,hs in d['hits'].items():
        if not os.path.exists(f'{rt.B}/players_v5/{k}.json'): continue
        N=load(k,1);O=OLD[k]
        for h in hs:
            if h['status'] in('not_a_hit','unsure_hit') : continue
            f=round(h['t']*N['fps']); got=None
            if h.get('hitter'):
                for e in (0,-1,1,-2,2,-3,3):
                    o=[p for p in O['by'].get(f+e,[]) if p.player_id==h['hitter']]
                    if not o: continue
                    b=max(N['by'].get(f+e,[]),key=lambda p:iou(p.bbox,o[0].bbox),default=None)
                    if b is not None and iou(b.bbox,o[0].bbox)>.4: got=b.player_id;break
            if got is None and N['fmt']=='singles': got=singles_hitter(N,h['t'],h['side'])
            c['ok' if got else 'none']+=1; h['hitter_old']=h.get('hitter');h['hitter']=got
    print(c); return d
json.dump(remap(src),open(rt.B+'/labels/amateur_v3_newids.json','w'),indent=1);json.dump(remap(v4),open(rt.B+'/labels/amateur_v4_newids.json','w'),indent=1)
