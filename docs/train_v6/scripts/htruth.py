"""htruth.py: pixel-anchored hitter truth for the labelled doubles hits -> labels/hitter_px.json
labelled id -> that player's v5 box at the hit; hitter-less labels -> my pick from review/hl/none_* (DEC below)."""
import json,os,sys
sys.path.insert(0,'/Users/mac/Desktop/RallyIQ/train_v5/scripts'); import rt
B6=os.path.expanduser('~/Desktop/RallyIQ/train_v6')
# (cand index | 'nodet' box | None=ambiguous, reason)
DEC={'own_20260310_01':{9:(None,'far pair overlap at the tape, cannot tell P1/P3'),11:(3,'far right player, shuttle on his racket'),15:(0,'near server (light trousers), shuttle at his racket'),
 19:(1,'near partner (light trousers) intercepts at the net'),23:(1,'near player at the net, shuttle at his head height'),26:(2,'far lunging player, shuttle at his racket'),
 28:(2,'far crouching player at the net, shuttle at racket'),32:(2,'far lunging player right, shuttle at racket'),33:(0,'near light-trousers player drives')},
 'own_20260324_0241':{3:(0,'near girl jumping overhead'),11:(3,'near boy at the net'),13:(1,'near boy lunges right to the shuttle'),14:(8,'far left player at the net, shuttle at racket'),
 17:([1040,520,1120,810],'near boy hidden behind the girl (no detection at the hit frame)'),19:(1,'near girl jumping overhead'),21:(0,'near girl, racket by the shuttle (medium)'),
 23:(5,'near boy, shuttle at him'),24:(5,'far left partner, racket up at the shuttle'),25:(8,'near boy lifts'),26:(6,'far boy behind P4 swings overhead (detected, untracked)'),
 27:(1,'near boy low backhand'),28:(1,'near boy jumping overhead'),33:(8,'far left player overhead')},
 'own_20260906_0622':{3:(3,'far right partner overhead'),4:(3,'far right player lunges'),13:([1515,332,1636,522],'far jumper hidden behind P1 (no detection at the hit frame)'),
 14:(0,'near left partner forehand'),19:(2,'near left partner low reach'),21:(1,'near left partner forehand'),46:(2,'far right partner overhead')}}
# label-id truths I overruled after viewing review/hl/chk_* (the v4->v5 id remap or the label was wrong)
OVR={'own_20260324_0241':{2:(4,'far girl at the net, shuttle at her racket (label P4 = far boy, too far)'),4:(3,'far girl at the net, shuttle at her racket'),
  18:(4,'far girl at the net beside the shuttle (label id mapped to the wrong far player)'),29:(4,'far boy, shuttle at his racket (label = far girl, 1 m away)')},
 'own_20260310_01':{5:(None,'shuttle between far P4 and P2, label itself unsure')},'own_20260320_0239':{2:(None,'hitter hidden behind the near player, label unsure')},
 'own_20260906_0622':{1:(3,'far right player jumping overhead, shuttle at racket'),16:(1,'near blue-shirt player overhead reach (label id mapped to the standing partner)')}}
P={}
out={}
for k in ['own_20260310_01','own_20260320_0239','own_20260324_0241','own_20260906_0622']:
    M=json.load(open(f'{B6}/review/hl/{k}.json')); T=json.load(open(f'{rt.B}/players_v5/{k}.json')); PB={f['frame']:f['players'] for f in T['frames']}
    rows=[]
    for m in M:
        r={'t':m['t'],'frame':m['frame'],'side':m['side'],'label':m['label']}
        if m['n'] in OVR.get(k,{}):
            c,why=OVR[k][m['n']]; r['bbox']=None if c is None else m['cand'][c]['bbox']; r['src']=('overruled: ' if c is not None else 'ambiguous: ')+why
        elif m['label'] is not None:
            for e in (0,-2,2,-4,4):
                b=[p['bbox'] for p in PB.get(m['frame']+e,[]) if p['player_id']==m['label']]
                if b: r['bbox']=b[0]; r['src']='label id (v5 box)'; break
            else: r['bbox']=None; r['src']='label id has no v5 box'
        else:
            c,why=DEC[k][m['n']]
            if c is None: r['bbox']=None; r['src']='ambiguous: '+why
            elif isinstance(c,list): r['bbox']=c; r['src']='nodet: '+why
            else: r['bbox']=m['cand'][c]['bbox']; r['src']='my pick: '+why
        rows.append(r)
    out[k]=rows; print(k,len(rows),sum(r['bbox'] is None for r in rows))
json.dump(out,open(f'{B6}/labels/hitter_px.json','w'),indent=0)
