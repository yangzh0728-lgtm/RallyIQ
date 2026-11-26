"""Rows of train_v4/review/final (same order as render_final.py) -> list of dicts with key,t,hitter,side,next_t,title,sheet,row."""
import json,os
B=os.path.expanduser('~/Desktop/RallyIQ/train_v4')
def items():
    its={}
    for rnd in ('r4','r4b','r4c'):
        cand=json.load(open(f'{B}/labels/{rnd}/candidates.json'))
        for l in open(f'{B}/labels/{rnd}/decisions.txt'):
            if l.startswith('#') or not l.strip(): continue
            key,idx,t,status,hitter,side,stroke,ss,note=[x.strip() for x in l.split('|',8)]
            c=cand['cands'][key][int(idx)]
            lab={'hit':'HIT','not_a_hit':'NOT A HIT','unsure_hit':'UNSURE'}[status]
            if status=='hit': lab+=f" {hitter}/{side} {stroke if stroke!='-' else 'stroke ?'}{' (sure)' if ss=='true' else ''}"
            its.setdefault(key,[]).append({'key':key,'t':c['t'],'hitter':hitter if hitter!='null' else None,'side':side,'id':f'{rnd}#{idx}','label':lab,'note':note})
    s1={it['n']:it for it in json.load(open(f'{B}/labels/strokes1_items.json'))}
    for l in open(f'{B}/labels/strokes1_decisions.txt'):
        if not l.startswith('S'): continue
        sid,st,sure,note=[x.strip() for x in l.split('|',3)]; it=s1[int(sid[1:])]
        its.setdefault(it['key'],[]).append({'key':it['key'],'t':it['t'],'hitter':it['hitter'],'side':it['side'],'next_t':it['next_t'],'id':sid,'label':f"STROKE {st if st!='-' else '?'}{' (sure)' if sure=='true' else ''}",'note':note})
    for key,v in its.items():
        v.sort(key=lambda x:x['t'])
        for i,it in enumerate(v):
            if it.get('next_t') is None:
                nx=[x['t'] for x in v[i+1:] if .25<x['t']-it['t']<3]; it['next_t']=nx[0] if nx else None
            it['sheet']=f'{key}_{i//4:02d}';it['row']=i%4+1
    return its
