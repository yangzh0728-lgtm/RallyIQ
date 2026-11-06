from rallyiq.pipeline import runner
from rallyiq.store import Store


def _video(store, sample_video):
    vid, dest = store.create_video("clip.mp4", ".mp4")
    dest.write_bytes(sample_video.read_bytes())
    store.register_video(vid, "clip.mp4", dest)
    return vid


def test_stub_pipeline_runs_without_demo(tmp_path, sample_video, monkeypatch):
    monkeypatch.delenv("RALLYIQ_DEMO", raising=False)
    store = Store(tmp_path)
    vid = _video(store, sample_video)
    job = runner.run_job(store, store.create_job(vid))
    assert job.state == "succeeded"
    assert store.has_artifact(vid, "overlay")


def test_stage_failure_fails_job_and_skips_rest(tmp_path, sample_video, monkeypatch):
    store = Store(tmp_path)
    vid = _video(store, sample_video)

    def boom(ctx):
        raise RuntimeError("model missing")

    real = runner.load_stage
    monkeypatch.setattr(runner, "load_stage", lambda n: boom if n == "shuttle" else real(n))
    job = runner.run_job(store, store.create_job(vid))
    states = {s.name: s.state for s in job.stages}
    assert job.state == "failed" and "model missing" in job.error
    assert states["players"] == "succeeded" and states["shuttle"] == "failed"
    assert states["events"] == "skipped" and states["overlay"] == "skipped"


def test_claim_is_exclusive(tmp_path, sample_video):
    store = Store(tmp_path)
    vid = _video(store, sample_video)
    store.create_job(vid)
    assert store.claim_next_job() is not None
    assert store.claim_next_job() is None
