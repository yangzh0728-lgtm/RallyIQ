import json

import cv2
import numpy as np

from rallyiq.contracts.geometry import COURT_CORNERS_M, fit_homography
from scripts import review_frames as rf


def test_review_frames_draws_court_players_and_shuttle(tmp_path):
    clip = tmp_path / "c.mp4"
    vw = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"), 30, (640, 360))
    for _ in range(30):
        vw.write(np.full((360, 640, 3), 40, np.uint8))
    vw.release()
    corners = [[100, 340], [540, 340], [420, 120], [220, 120]]
    h = fit_homography(corners, COURT_CORNERS_M)  # image -> court, like CalibrationSegment
    art = tmp_path / "art"
    art.mkdir()
    (art / "calibration.json").write_text(json.dumps(
        {"segments": [{"start_frame": 0, "end_frame": 29, "homography": h, "status": "ok"}]}))
    (art / "players.json").write_text(json.dumps({"frames": [
        {"frame": 10, "players": [{"track_id": 1, "player_id": "P1", "side": "near", "bbox": [300, 200, 340, 330],
                                   "confidence": 0.9, "on_court": True}]}]}))
    (art / "shuttle.json").write_text(json.dumps({"points": [{"frame": f, "x": 320 + f, "y": 100, "confidence": 0.8}
                                                             for f in range(5, 11)]}))
    (art / "events.json").write_text(json.dumps({"shots": [{"frame": 10}], "rallies": [{"start_frame": 2, "end_frame": 20}]}))
    rc = rf.main(["--clip", str(clip), "--artifacts", str(art), "--out", str(tmp_path / "out"), "--frames", "10", "25"])
    assert rc == 0
    s = json.loads((tmp_path / "out" / "c_summary.json").read_text())
    assert s["court_frames"] == 30 and s["hits"] == 1 and len(s["stills"]) == 2
    assert s["shuttle_in_rally_frac"] == round(6 / 19, 3)
    img = cv2.imread(str(tmp_path / "out" / "c_f00010.jpg"))
    b, g, r = img[340, 320].astype(int)  # near baseline is magenta
    assert r > 150 and b > 150 and g < 120
    assert (tmp_path / "out" / "c_sheet.jpg").exists()
