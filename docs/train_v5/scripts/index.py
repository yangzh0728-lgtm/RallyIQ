import sys,json,os,collections; sys.path.insert(0,'/Users/mac/Desktop/RallyIQ/train_v5/scripts')
import rt,numpy as np
B=rt.B; ab=json.load(open(B+'/review/ab_counts.json'))
# old bystander rows: sheets of 0241/0618/0619/0620/0621 drew the old players; count rows whose AT HIT/NEXT frame had an old box past a sideline
import sheet_items,replay
old_c=collections.Counter()
for k,its in sheet_items.items().items():
    cal=replay.calib(k);t=rt.PlayerTracks.model_validate_json(rt.rd(rt.art(k),'players'));fps=float(rt.IDS[k][2]);P={f.frame:f.players for f in t.frames}
    for it in its:
        bad=False
        for q in (it['t'],min(it.get('next_t') or it['t']+1.5,it['t']+2.5)):
            f=round(q*fps)
            for d in (0,-1,1,-2,2):
                for p in P.get(f+d,[]):
                    if p.player_id and p.foot_px:
                        H=np.array(cal.segment_for(f).homography);v=H@[*p.foot_px,1];x=v[0]/v[2]
                        if max(-x,x-6.1)>.6: bad=True
                if P.get(f+d): break
        old_c[it['sheet']]+=bad
rows=[];tot=collections.Counter()
for l in open(B+'/review/final/check_v5.txt'):
    if not l.startswith('own'): continue
    s,a,b,c,m,note=[x.strip() for x in l.split('|',5)]
    oa=sum(1 for r in ab.get(s,[]) if r[2]=='a');ob=sum(1 for r in ab.get(s,[]) if r[2]=='b');oc=old_c[s]
    rows.append((s,oa,ob,oc,int(a),int(b),int(c),int(m),note))
    for n,v in zip(('oa','ob','oc','a','b','c','m'),(oa,ob,oc,int(a),int(b),int(c),int(m))): tot[n]+=v
rows.sort()
L=['# Review sheets, train_v5 (regenerated with the cleaned shuttle tracks and players)','',
'Each sheet: 4 rows (same rows as train_v4/review/final, same file names). Row = AT HIT overview with the trail of *this* shuttle',
'(blue before, orange after; it no longer jumps onto another shuttle), 3 crops of the hitter (-0.20 / 0.00 / +0.15 s), NEXT overview.',
'Green circle = tracked shuttle at that frame. Titles keep the label text of train_v4 (player ids in titles are the old ones).','',
'Columns: a = right shuttle at the hit but the trail includes another court\'s shuttle; b = the circle at AT HIT or NEXT is on another',
'court\'s shuttle; c = a spectator / bystander boxed as a player at AT HIT or NEXT; m = our shuttle missed at the hit.',
'"before" = the same rows drawn from the train_v4 tracks: a/b by matching to my hand-labelled next-court trajectories (264 rows),',
'c by any player box > 0.6 m outside a sideline at those frames. "after" = my own reading of every new sheet.','',
f"**Totals (68 sheets, 264 rows): a {tot['oa']} -> {tot['a']}, b {tot['ob']} -> {tot['b']}, c {tot['oc']} -> {tot['c']}; after only: m {tot['m']}.**",'',
'| sheet | a before | b before | c before | a after | b after | c after | m after | note |','|---|---|---|---|---|---|---|---|---|']
for r in rows: L.append('| '+' | '.join(map(str,r))+' |')
L+=['','Changed sheets: every sheet was regenerated; the ones whose before/after counts differ are those with a nonzero before column.',
'Remaining known issues (after): see rows with a/b/m > 0. The b rows left are all between rallies (nobody of ours hitting), and the',
'a rows are next-court arcs that pass between our two players, which the ownership test cannot tell from ours.']
open(B+'/review/final/INDEX.md','w').write('\n'.join(L)+'\n'); print(tot)
