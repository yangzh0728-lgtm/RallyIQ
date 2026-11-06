"""ShuttleSet (KDD 2023, CoachAI-Projects, MIT) -> evaluation labels for court, hits and rallies.

ShuttleSet annotates every stroke of 44 BWF singles matches: hit frame, hitter, player
positions, plus one court homography per match. Coordinates are 1280x720 pixels of the
YouTube broadcast; times are frame numbers of that video.

What it gives us:
- hits:    t_ms of every stroke + near/far side (hitter position mapped through the match homography)
- rallies: first stroke -> last stroke + RALLY_TAIL_MS (ShuttleSet has no landing frame, so the
           rally end is approximate; rally F1 uses IoU >= 0.5, which tolerates that)
- court:   the four court corners at every stroke time (strokes happen on the main camera view)
- shuttle: nothing (ShuttleSet has no per-frame shuttle positions)
"""
from __future__ import annotations

import ast
import csv
import statistics
from pathlib import Path
from typing import Any

import numpy as np

from .. import schema

SRC_W, SRC_H = 1280, 720
RALLY_TAIL_MS = 1000
STD_FPS = (25.0, 30.0, 50.0, 60.0)


def _secs(t: str) -> int:
    h, m, s = (int(x) for x in t.split(":"))
    return h * 3600 + m * 60 + s


def _f(v: str) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load_matches(set_dir: Path) -> list[dict[str, Any]]:
    """match.csv joined with homography.csv."""
    homs = {r["video"]: r for r in csv.DictReader(open(set_dir / "homography.csv", encoding="utf-8"))}
    out = []
    for r in csv.DictReader(open(set_dir / "match.csv", encoding="utf-8")):
        h = homs.get(r["video"])
        if h is None:
            continue
        r = dict(r)
        r["H"] = np.asarray(ast.literal_eval(h["homography_matrix"]), float)
        # ours: far-left, far-right, near-right, near-left (image "up" is the far side)
        r["corners"] = [[float(h["upleft_x"]), float(h["upleft_y"])], [float(h["upright_x"]), float(h["upright_y"])],
                        [float(h["downright_x"]), float(h["downright_y"])], [float(h["downleft_x"]), float(h["downleft_y"])]]
        r["youtube_id"] = r["url"].split("v=")[-1].split("&")[0]
        out.append(r)
    return out


def eval_matches(matches: list[dict[str, Any]], n: int = 6) -> list[dict[str, Any]]:
    """The held-out matches scored by scripts/public_eval.py (every k-th match). Never train on these."""
    step = max(1, len(matches) // n)
    return matches[::step][:n]


def train_matches(matches: list[dict[str, Any]], n_eval: int = 6) -> list[dict[str, Any]]:
    """Every match not held out for evaluation. The split is by match, so no player/venue/broadcast
    of an evaluation match leaks into training."""
    held = {m["id"] for m in eval_matches(matches, n_eval)}
    return [m for m in matches if m["id"] not in held]


def load_strokes(set_csv: Path) -> list[dict[str, Any]]:
    rows = []
    for r in csv.DictReader(open(set_csv, encoding="utf-8")):
        fn = _f(r.get("frame_num"))
        if fn is None or not r.get("rally"):
            continue
        rows.append({"rally": int(float(r["rally"])), "frame": fn, "sec": _secs(r["time"]), "player": r.get("player"),
                     "loc": (_f(r.get("player_location_x")), _f(r.get("player_location_y")))})
    return rows


def video_fps(strokes: list[dict[str, Any]]) -> float:
    est = statistics.median(s["frame"] / s["sec"] for s in strokes if s["sec"] > 0)
    return min(STD_FPS, key=lambda f: abs(f - est))


def _to_world(H: np.ndarray, xy: tuple[float, float]) -> np.ndarray:
    p = H @ np.array([xy[0], xy[1], 1.0])
    return p[:2] / p[2]


def side_of(match: dict[str, Any], loc: tuple[float | None, float | None]) -> str | None:
    """near/far of a player position: closer (in court coordinates) to the near or the far baseline."""
    if loc[0] is None or loc[1] is None:
        return None
    H, c = match["H"], match["corners"]
    far_y = np.mean([_to_world(H, c[0])[1], _to_world(H, c[1])[1]])
    near_y = np.mean([_to_world(H, c[2])[1], _to_world(H, c[3])[1]])
    y = _to_world(H, loc)[1]
    return "near" if abs(y - near_y) < abs(y - far_y) else "far"


def pick_window(strokes: list[dict[str, Any]], fps: float, max_s: float, first_rally: int = 3) -> tuple[float, float]:
    """[start_s, end_s] covering whole rallies, starting at rally `first_rally`, at most max_s long."""
    by_rally: dict[int, list[float]] = {}
    for s in strokes:
        by_rally.setdefault(s["rally"], []).append(s["frame"] / fps)
    rallies = sorted(by_rally.items())
    rallies = [r for r in rallies if r[0] >= first_rally] or rallies
    start = max(0.0, min(rallies[0][1]) - 2.0)
    end = start
    for _, ts in rallies:
        e = max(ts) + RALLY_TAIL_MS / 1000 + 0.5
        if e - start > max_s:
            break
        end = e
    if end <= start:  # first rally alone is longer than max_s
        end = start + max_s
    return start, end


def build_label(clip_id: str, match: dict[str, Any], strokes: list[dict[str, Any]], fps: float,
                window: tuple[float, float], width: int, height: int) -> dict[str, Any]:
    """Label doc in the clip's pixel space (ShuttleSet pixels are scaled from 1280x720)."""
    s0, s1 = window
    r0, r1 = int(round(s0 * 1000)), int(round(s1 * 1000))
    sx, sy = width / SRC_W, height / SRC_H
    doc = schema.empty_doc(clip_id, "label", width, height, fps, (r0, r1))
    doc["status"] = "reviewed"
    doc["source"] = {"dataset": "ShuttleSet", "video": match["video"], "youtube_id": match["youtube_id"]}
    corners = [[x * sx, y * sy] for x, y in match["corners"]]
    rallies: dict[int, list[int]] = {}
    for s in strokes:
        t = int(round(s["frame"] * 1000 / fps))
        if not r0 <= t <= r1:
            continue
        rallies.setdefault(s["rally"], []).append(t)
        doc["hits"].append({"t_ms": t, "side": side_of(match, s["loc"]), "player": None})
        doc["court"].append({"t_ms": t, "visible": True, "corners": corners})
    for ts in rallies.values():
        doc["rallies"].append({"start_ms": min(ts), "end_ms": min(max(ts) + RALLY_TAIL_MS, r1)})
    return schema.validate(doc)


def manifest_entry(clip_id: str, match: dict[str, Any], window: tuple[float, float]) -> dict[str, Any]:
    return {"id": clip_id, "video_id": match["youtube_id"], "title": match["video"],
            "start_s": round(window[0], 3), "duration_s": round(window[1] - window[0], 3),
            "segment_verified": True, "split": "eval",
            "conditions": {"event": match.get("tournament"), "venue": match.get("tournament"),
                           "discipline": "singles", "camera": "broadcast_main", "source": "shuttleset"}}
