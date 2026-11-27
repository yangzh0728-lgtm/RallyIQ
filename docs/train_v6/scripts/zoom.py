"""zoom.py CLIP ROI(x0,y0,x1,y1|full) THR: detector candidates on a clip with an ROI -> prints per frame."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path('~/Desktop/RallyIQ/train_v6/repo').expanduser()))
from rallyiq.perception.shuttle.detector import HeatmapShuttleDetector,DetectorConfig,Segment,find_weights
clip=Path(sys.argv[1]); roi=None if sys.argv[2]=='full' else tuple(int(v) for v in sys.argv[2].split(',')); thr=float(sys.argv[3])
d=HeatmapShuttleDetector(Path('~/RallyIQ/weights/shuttle/heatmap.pt').expanduser(),DetectorConfig(peak_threshold=thr,top_k=5,target_fps=0))
import cv2; n=int(cv2.VideoCapture(str(clip)).get(7)); fps=cv2.VideoCapture(str(clip)).get(5)
res,_=d.detect(clip,fps,n,[Segment(0,n-1,roi)])
json.dump({int(k):v for k,v in res.items()},open(str(clip)+f'.{sys.argv[2]}.json','w'))
for f in sorted(res):
    if res[f]: print(f,[(round(x),round(y),round(s,2)) for x,y,s in res[f]])
