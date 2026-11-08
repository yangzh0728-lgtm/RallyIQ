from pathlib import Path

from rallyiq.contracts import ShuttleTrack, VideoMeta
from rallyiq.perception.shuttle import stage
from rallyiq.perception.shuttle.export import to_benchmark_prediction
from rallyiq.pipeline.context import StageContext
from rallyiq.store import Store


def _meta():
    return VideoMeta(video_id="v1", filename="a.mp4", fps=25.0, frame_count=50, width=1920, height=1080,
                     duration_ms=2000)


def test_no_weights_returns_unanalysed_track(tmp_path, monkeypatch):
    monkeypatch.delenv("RALLYIQ_DEMO", raising=False)
    monkeypatch.setattr("rallyiq.perception.shuttle.detector.find_weights", lambda explicit=None: None)
    ctx = StageContext(video_id="v1", video_path=Path("missing.mp4"), work_dir=tmp_path, store=Store(tmp_path),
                       meta=_meta())
    track = stage.run(ctx)
    assert track.detector is None and track.points == []

    ctx.config = {"shuttle": {"require_weights": True}}
    try:
        stage.run(ctx)
    except RuntimeError as e:
        assert "weights" in str(e)
    else:
        raise AssertionError("expected RuntimeError")


def test_benchmark_export_excludes_interpolated(tmp_path):
    meta = _meta()
    cands = {f: [(100.0 + 10 * f, 500.0 - 5 * f, 0.9)] for f in range(10, 30) if f not in (15, 16)}
    track, _ = stage.build_track(meta, cands, 1920, 1080, "test@0")
    assert any(p.source == "interpolated" for p in track.points)
    doc = to_benchmark_prediction(track, meta, "clip1", start_ms=900000)
    vis = [s for s in doc["shuttle"] if s["visible"]]
    assert len(vis) == 18
    times = {s["t_ms"] for s in doc["shuttle"]}
    assert 900000 + 15 * 40 not in times  # interpolated frame: neither visible nor invisible
    assert 900000 in times and doc["video"]["fps"] == 25.0
    assert doc["range_ms"] == [900000, 902000]


def test_candidates_file_keeps_rejected(tmp_path):
    meta = _meta()
    cands = {f: [(100.0 + 10 * f, 500.0, 0.9), (1800.0, 50.0, 0.3)] for f in range(0, 20)}
    track, selected = stage.build_track(meta, cands, 1920, 1080, "test@0")
    ShuttleTrack.model_validate(track.model_dump())
    out = tmp_path / "c.json"
    stage.save_candidates(out, cands, selected, "test@0", (1920, 1080))
    import json

    doc = json.loads(out.read_text())
    assert doc["frames"]["5"]["candidates"][1] == [1800.0, 50.0, 0.3]
    assert doc["frames"]["5"]["selected"] == 0


def test_cli_retracks_saved_candidates_without_model(tmp_path, sample_video):
    from rallyiq.perception.shuttle.__main__ import main

    # sample_video: 640x360, 30 fps, 12 s
    cands = {f: [(100.0 + 3 * f, 300.0 - 2 * f, 0.15), (500.0, 50.0, 0.15)] for f in range(0, 60)}
    cfile = tmp_path / "c.json"
    stage.save_candidates(cfile, cands, {}, "test@0", (640, 360))
    out = tmp_path / "s.json"
    assert main([str(sample_video), "--from-candidates", str(cfile), "--peak-threshold", "0.1",
                 "--out", str(out)]) == 0
    track = ShuttleTrack.model_validate_json(out.read_text())
    assert track.detector == "test@0"
    xs = {p.frame: p.x for p in track.points if p.source == "detected"}
    assert len(xs) >= 55 and abs(xs[30] - 190.0) < 1e-6  # the weak moving track, not the idle blob
    # Default threshold (0.2): everything is below it; an explicit weak_score override wins.
    assert main([str(sample_video), "--from-candidates", str(cfile), "--out", str(out)]) == 0
    assert ShuttleTrack.model_validate_json(out.read_text()).points == []
    assert main([str(sample_video), "--from-candidates", str(cfile), "--out", str(out), "--peak-threshold", "0.1",
                 "--tracker", "weak_score=0.5"]) == 0
    assert ShuttleTrack.model_validate_json(out.read_text()).points == []


def test_checkpoint_peak_threshold_is_used_unless_configured(tmp_path):
    import pytest

    torch = pytest.importorskip("torch")
    from rallyiq.perception.shuttle.model import build_heatmap_net

    w = tmp_path / "ft.pt"
    torch.save({"model": build_heatmap_net(3, "concat").state_dict(),
                "param_dict": {"seq_len": 3, "bg_mode": "concat", "peak_threshold": 0.3}}, w)
    ctx = StageContext(video_id="v1", video_path=Path("missing.mp4"), work_dir=tmp_path, store=Store(tmp_path),
                       meta=_meta())
    ctx.config = {"shuttle": {"weights": str(w), "device": "cpu"}}
    det, cfg = stage._detector(ctx)
    assert cfg.peak_threshold == 0.3 and det.cfg.peak_threshold == 0.3
    ctx.config = {"shuttle": {"weights": str(w), "device": "cpu", "peak_threshold": 0.2}}
    _det, cfg = stage._detector(ctx)
    assert cfg.peak_threshold == 0.2  # an explicit setting wins
