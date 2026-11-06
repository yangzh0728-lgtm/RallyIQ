"""Shuttlecock Trajectory Dataset (TrackNetV2/V3) -> evaluation labels for the shuttle.

Layout after extracting the dataset:  <root>/<split>/<match>/csv/<rally>_ball.csv  and
<root>/<split>/<match>/video/<rally>.mp4.  Each csv has Frame, Visibility, X, Y per frame,
in the pixels of the rally video.

Caveat: our default shuttle checkpoint is TrackNetV3 trained on this dataset's train split.
Scores on its test split are held-out matches from the same broadcasts, so they are an
in-domain upper bound, not what to expect on phone footage or side cameras.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .. import schema


def rallies(root: Path, split: str = "test") -> list[tuple[str, Path, Path]]:
    """(clip_id, video, csv) for every rally of a split that has both files."""
    out = []
    for c in sorted((root / split).glob("*/csv/*_ball.csv")):
        match = c.parent.parent.name
        rally = c.name[: -len("_ball.csv")]
        v = c.parent.parent / "video" / f"{rally}.mp4"
        if v.exists():
            out.append((f"tn_{split}_{match}_{rally}", v, c))
    return out


def build_label(clip_id: str, csv_path: Path, width: int, height: int, fps: float,
                frame_count: int) -> dict[str, Any]:
    dur = int(round(max(frame_count, 1) * 1000 / fps))
    doc = schema.empty_doc(clip_id, "label", width, height, fps, (0, dur))
    doc["status"] = "reviewed"
    doc["source"] = {"dataset": "Shuttlecock Trajectory Dataset", "csv": csv_path.name}
    for r in csv.DictReader(open(csv_path, encoding="utf-8")):
        f = int(float(r["Frame"]))
        vis = int(float(r["Visibility"])) == 1
        doc["shuttle"].append({"t_ms": int(round(f * 1000 / fps)), "visible": vis,
                               "xy": [float(r["X"]), float(r["Y"])] if vis else None})
    return schema.validate(doc)
