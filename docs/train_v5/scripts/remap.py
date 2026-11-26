"""Rewrite label hitter ids onto the replayed players (players_v5): match the labelled player's old box to the new box
(IoU) around the hit frame; singles: the only player on the labelled side. Labels without a hitter stay without one."""
import sys,os,json,copy; sys.path.insert(0,'/Users/mac/Desktop/RallyIQ/train_v5/scripts')
import rt
from collections import Counter
def iou(a,b):
    w=min(a[2],b[2])-max(a[0],b[0]);h=min(a[3],b[3])-max(a[1],b[1])
    if w<=0 or h<=0: return 0
    i=w*h; return i/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-i)
def by(t): return {f.frame:[p for p in f.players if p.player_id and not p.interpolated] for f in t.frames}
if __name__=='__main__':
  for name in ('amateur_v3','amateur_v4'):
      src=json.load(open(rt.H+'/train_v4/labels/amateur_v3.json') if name=='amateur_v3' else open(rt.B+'/labels/amateur_v4.json'));d=copy.deepcopy(src);c=Counter()
      for k,hs in d['hits'].items():
          p=f'{rt.B}/players_v5/{k}.json'
          if not os.path.exists(p): continue
          fps=float(rt.IDS[k][2]);N=rt.PlayerTracks.model_validate_json(open(p).read());O=by(rt.PlayerTracks.model_validate_json(rt.rd(rt.art(k),'players')));NB=by(N)
          for h in hs:
              if h['status'] not in('reviewed_sure','reviewed_unsure') or not h.get('hitter'): continue
              f=round(h['t']*fps);got=None
              for e in (0,-1,1,-2,2,-3,3,-4,4):
                  o=[q for q in O.get(f+e,[]) if q.player_id==h['hitter']]
                  if o and NB.get(f+e):
                      b=max(NB[f+e],key=lambda q:iou(q.bbox,o[0].bbox))
                      if iou(b.bbox,o[0].bbox)>.4: got=b.player_id;break
              if got is None and N.match_format=='singles':
                  for e in (0,-1,1,-2,2,-3,3,-4,4):
                      s=[q for q in NB.get(f+e,[]) if q.side==h['side']]
                      if len(s)==1: got=s[0].player_id;break
              c['ok' if got else 'none']+=1;h['hitter_v4']=h['hitter'];h['hitter']=got
      json.dump(d,open(f'{rt.B}/labels/{name}_newids.json','w'),indent=1);print(name,c)
