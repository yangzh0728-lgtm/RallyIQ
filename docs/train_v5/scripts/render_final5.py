import sys,os; sys.path.insert(0,'/Users/mac/Desktop/RallyIQ/train_v5/scripts'); os.environ.setdefault('INP','inputs_C')
import sheet_items,sheets5 as S,rt
from multiprocessing import Pool
from remap import iou,by
def newid(k,its):
    p=f'{rt.B}/players_v5/{k}.json'
    if not os.path.exists(p): return
    fps=float(rt.IDS[k][2]);O=by(rt.PlayerTracks.model_validate_json(rt.rd(rt.art(k),'players')));N=by(rt.PlayerTracks.model_validate_json(open(p).read()))
    for it in its:
        if not it.get('hitter'): continue
        f=round(it['t']*fps);got=None
        for e in (0,-1,1,-2,2,-3,3,-4,4):
            o=[q for q in O.get(f+e,[]) if q.player_id==it['hitter']]
            if o and N.get(f+e):
                b=max(N[f+e],key=lambda q:iou(q.bbox,o[0].bbox))
                if iou(b.bbox,o[0].bbox)>.4: got=b.player_id;break
        it['hitter']=got
def go(a):
    key,its=a
    newid(key,its)
    for it in its: it['title']=f"{it['id']} t={it['t']:.2f} {it['label']} | {it['note'][:80]}"
    return key,len(S.render(its,f'{rt.B}/review/final',key,per=4))
if __name__=='__main__':
    only=sys.argv[1:]
    with Pool(4) as p:
        for r in p.imap(go,[(k,v) for k,v in sheet_items.items().items() if not only or k in only]): print(*r,flush=True)
