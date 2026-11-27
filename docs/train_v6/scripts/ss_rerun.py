"""ss_rerun.py NAME PEAK: rebuild the 6 ShuttleSet clips' tracks from ckpt/b/pub_NAME/ss/cands at peak threshold PEAK and score hits."""
import os,sys,json,subprocess
from pathlib import Path
R=Path('~/Desktop/RallyIQ/train_v6/repo').expanduser(); os.environ['RTREPO']=str(R)
sys.path.insert(0,'/Users/mac/Desktop/RallyIQ/train_v5/scripts'); import rt
name,peak=sys.argv[1],float(sys.argv[2]); D=Path(f'~/Desktop/RallyIQ/train_v6/ckpt/b/pub_{name}/ss').expanduser()
for e in json.loads((D/'results.json').read_text()):
    od=Path(e['artifacts']); c=rt.Ctx(e['clip_id'],str(od),str(D/'cands'/f"{e['clip_id']}.json")); tr,_=c.run(peak=peak); (od/'shuttle.json').write_text(tr.model_dump_json())
ev=subprocess.run([sys.executable,str(R/'scripts/learn_hits.py'),'eval','--data',str(D),'--model',str(R/'rallyiq/events/models/hit_scorer.json'),'--out-name',f'e{peak}'],capture_output=True,text=True,cwd=str(R))
print(name,peak,'\n'.join(l for l in ev.stdout.splitlines() if l.startswith('| rules') or l.startswith('| hit_scorer')))
