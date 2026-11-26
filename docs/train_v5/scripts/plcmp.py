import sys,json; sys.path.insert(0,'/Users/mac/Desktop/RallyIQ/train_v5/scripts')
import rt,collections
k=sys.argv[1]; fps=float(rt.IDS[k][2])
O=rt.PlayerTracks.model_validate_json(rt.rd(rt.art(k),'players'));N=rt.PlayerTracks.model_validate_json(open(f'{rt.B}/players_v5/{k}.json').read())
L=[h for h in json.load(open(rt.B+'/labels/amateur_v4.json'))['hits'][k] if h['status'].startswith('reviewed')]
def at(T,f):
    m={x.frame:x for x in T.frames}
    for d in (0,-1,1,-2,2): 
        if f+d in m: return m[f+d].players
    return []
c=collections.Counter()
for h in L[:400]:
    f=round(h['t']*fps); o=at(O,f); n=at(N,f)
    so=collections.Counter(p.side for p in o if p.player_id); sn=collections.Counter(p.side for p in n if p.player_id)
    c['old_hitside_present']+=so[h['side']]>0; c['new_hitside_present']+=sn[h['side']]>0; c['n']+=1
    c['old_kp']+=sum(1 for p in o if p.player_id and p.side==h['side'] and sum(1 for q in p.keypoints if q[2]>.3)>=10)
    c['new_kp']+=sum(1 for p in n if p.player_id and p.side==h['side'] and sum(1 for q in p.keypoints if q[2]>.3)>=10)
print(k,O.match_format,N.match_format,dict(c))
p0=O.frames[len(O.frames)//2].players[0]; p1=N.frames[len(N.frames)//2].players[0]; print(p0.keypoints[:3],p1.keypoints[:3],p0.confidence,p1.confidence)
