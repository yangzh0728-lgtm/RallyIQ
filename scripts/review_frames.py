"""Draw what the pipeline found onto clip frames, for a quick visual check before labeling.

    python scripts/review_frames.py --clip ~/rallyiq-data/bwf/clips/own0529_0006.mp4 \\
        --artifacts ~/rallyiq-data/own/run_0529/data/videos/<id>/artifacts --out /tmp/review

Each still shows the calibrated court lines (magenta), the net (yellow), every player box with
its side (near = blue, far = orange, off court = grey), the shuttle (green = detected, yellow =
interpolated) with its last half second of track, and a red "HIT" tag on frames within 3 frames
of a detected hit. Frames are picked evenly over the clip plus the frames of the first hits,
so both quiet and busy moments get looked at. A ``summary.json`` gives the per-clip coverage
numbers the stills are judged against, and ``sheet.jpg`` puts every still on one page.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rallyiq.contracts.geometry import apply_homography, invert_homography

COURT_W, COURT_L, NET_Y = 6.10, 13.40, 6.70
_X = (0.0, 0.46, 3.05, 5.64, 6.10)
_Y = (0.0, 0.76, 4.72, 8.68, 12.64, 13.40)


def court_segments() -> list[tuple[tuple[float, float], tuple[float, float]]]:
    segs = [((0.0, y), (COURT_W, y)) for y in _Y]
    segs += [((x, 0.0), (x, COURT_L)) for x in (0.0, 0.46, 5.64, 6.10)]
    segs += [((3.05, 0.0), (3.05, 4.72)), ((3.05, 8.68), (3.05, COURT_L))]
    return segs


def _read(d: Path, name: str) -> dict[str, Any] | None:
    p = d / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def load(art: Path) -> dict[str, Any]:
    cal = _read(art, "calibration") or {}
    players = _read(art, "players") or {}
    shuttle = _read(art, "shuttle") or {}
    events = _read(art, "events") or {}
    return {
        "segments": cal.get("segments", []),
        "players": {fr["frame"]: fr["players"] for fr in players.get("frames", [])},
        "shuttle": {p["frame"]: p for p in shuttle.get("points", [])},
        "hits": sorted(s["frame"] for s in events.get("shots", [])),
        "rallies": [(r["start_frame"], r["end_frame"]) for r in events.get("rallies", [])],
    }


def segment_for(segments: list[dict[str, Any]], frame: int) -> dict[str, Any] | None:
    for s in segments:
        if s["start_frame"] <= frame <= s["end_frame"] and s.get("homography") and s.get("status") != "undeterminable":
            return s
    return None


def draw_court(img: np.ndarray, seg: dict[str, Any]) -> None:
    to_img = invert_homography(seg["homography"])
    for a, b in court_segments():
        pts = [[a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t] for t in np.linspace(0, 1, 40)]
        p = np.asarray(apply_homography(to_img, pts), dtype=np.float32)
        cv2.polylines(img, [p.astype(np.int32)], False, (255, 0, 255), 2, cv2.LINE_AA)
    net = np.asarray(apply_homography(to_img, [[0.0, NET_Y], [COURT_W, NET_Y]])).astype(int)
    cv2.line(img, tuple(net[0]), tuple(net[1]), (0, 255, 255), 2, cv2.LINE_AA)


def draw_frame(img: np.ndarray, frame: int, fps: float, data: dict[str, Any]) -> np.ndarray:
    seg = segment_for(data["segments"], frame)
    if seg is not None:
        draw_court(img, seg)
    for p in data["players"].get(frame, []):
        x1, y1, x2, y2 = map(int, p["bbox"])
        col = {"near": (255, 140, 0), "far": (0, 140, 255)}.get(p.get("side"), (160, 160, 160))
        if p.get("on_court") is False:
            col = (160, 160, 160)
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
        tag = f"{p.get('player_id') or '#' + str(p.get('track_id'))} {p.get('side') or '?'} {p.get('confidence', 0):.2f}"
        cv2.putText(img, tag, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
    trail = max(2, round(fps / 2))
    prev = None
    for f in range(frame - trail, frame + 1):
        pt = data["shuttle"].get(f)
        if pt is None:
            prev = None
            continue
        xy = (int(pt["x"]), int(pt["y"]))
        if prev is not None:
            cv2.line(img, prev, xy, (0, 200, 0), 1, cv2.LINE_AA)
        prev = xy
    cur = data["shuttle"].get(frame)
    if cur is not None:
        col = (0, 255, 0) if cur.get("source", "detected") == "detected" else (0, 255, 255)
        cv2.circle(img, (int(cur["x"]), int(cur["y"])), 12, col, 2, cv2.LINE_AA)
    if any(abs(h - frame) <= 3 for h in data["hits"]):
        cv2.putText(img, "HIT", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 255), 4, cv2.LINE_AA)
    in_rally = any(a <= frame <= b for a, b in data["rallies"])
    head = (f"f{frame}  t={frame / fps:6.2f}s  court={'ok' if seg else 'NONE'}  "
            f"shuttle={'yes' if cur else 'no'}  players={len(data['players'].get(frame, []))}"
            f"  {'rally' if in_rally else '-'}")
    cv2.rectangle(img, (0, 0), (img.shape[1], 50), (0, 0, 0), -1)
    cv2.putText(img, head, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def pick_frames(n_frames: int, hits: list[int], n_even: int, n_hits: int) -> list[int]:
    even = [round(i * (n_frames - 1) / max(n_even - 1, 1)) for i in range(n_even)]
    step = max(1, len(hits) // max(n_hits, 1))
    return sorted(set(even + hits[::step][:n_hits]))


def summary(data: dict[str, Any], n: int) -> dict[str, Any]:
    rally = {f for a, b in data["rallies"] for f in range(a, b + 1)}
    det = {f for f, p in data["shuttle"].items() if p.get("source", "detected") == "detected"}
    on = [sum(1 for p in data["players"].get(f, []) if p.get("on_court") is not False) for f in range(n)]
    return {
        "frames": n,
        "court_frames": sum(1 for f in range(n) if segment_for(data["segments"], f)),
        "shuttle_detected_frac": round(len(det) / n, 3) if n else None,
        "shuttle_in_rally_frac": round(len(det & rally) / len(rally), 3) if rally else None,
        "players_two_on_court_frac": round(sum(1 for c in on if c >= 2) / n, 3) if n else None,
        "rallies": len(data["rallies"]),
        "hits": len(data["hits"]),
    }


def contact_sheet(paths: list[Path], out: Path, cols: int = 4, width: int = 480) -> None:
    tiles = []
    for p in paths:
        im = cv2.imread(str(p))
        tiles.append(cv2.resize(im, (width, int(im.shape[0] * width / im.shape[1]))))
    if not tiles:
        return
    h = tiles[0].shape[0]
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [cv2.hconcat(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    cv2.imwrite(str(out), cv2.vconcat([r[:h] for r in rows]), [cv2.IMWRITE_JPEG_QUALITY, 85])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", type=Path, required=True)
    ap.add_argument("--artifacts", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--even", type=int, default=12, help="evenly spaced frames")
    ap.add_argument("--hits", type=int, default=8, help="frames at detected hits")
    ap.add_argument("--frames", type=int, nargs="*", help="exact frames instead of the automatic pick")
    a = ap.parse_args(argv)

    data = load(a.artifacts.expanduser())
    cap = cv2.VideoCapture(str(a.clip.expanduser()))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = a.frames or pick_frames(n, data["hits"], a.even, a.hits)
    out = a.out.expanduser()
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for f in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if not ok:
            continue
        p = out / f"{a.clip.stem}_f{f:05d}.jpg"
        cv2.imwrite(str(p), draw_frame(img, f, fps, data), [cv2.IMWRITE_JPEG_QUALITY, 88])
        written.append(p)
    contact_sheet(written, out / f"{a.clip.stem}_sheet.jpg")
    s = summary(data, n) | {"clip": a.clip.stem, "stills": [p.name for p in written]}
    (out / f"{a.clip.stem}_summary.json").write_text(json.dumps(s, indent=1), encoding="utf-8")
    print(json.dumps(s, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
