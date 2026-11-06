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


def _prefetch_stage(monkeypatch, started, players_fn=None):
    """Make ``shuttle`` start alongside ``players`` with a fake background job."""
    import threading

    from rallyiq.pipeline import registry
    from rallyiq.contracts import ShuttleTrack

    def prefetch(ctx, report, cancel):
        done = threading.Event()

        def work():
            started.append("prefetch")
            report(0.5)
            cancel.wait(0.05)
            done.set()

        threading.Thread(target=work, daemon=True).start()

        def wait():
            done.wait(5)
            started.append("waited")
            return "candidates"

        return wait

    def shuttle(ctx):
        got = ctx.prefetched.pop("shuttle")()
        started.append(f"shuttle:{got}")
        return ShuttleTrack(video_id=ctx.video_id, detector=None, points=[])

    real = runner.load_stage
    monkeypatch.setattr(runner, "load_prefetch", lambda n: prefetch)
    monkeypatch.setattr(runner, "load_stage", lambda n: shuttle if n == "shuttle" else
                        (players_fn if n == "players" and players_fn else real(n)))
    assert "shuttle" in registry.STAGE_PREFETCH


def test_shuttle_starts_alongside_players(tmp_path, sample_video, monkeypatch):
    store = Store(tmp_path)
    vid = _video(store, sample_video)
    log = []
    real_players = runner.load_stage("players")

    def players(ctx):
        log.append("players")
        return real_players(ctx)

    _prefetch_stage(monkeypatch, log, players)
    job = runner.run_job(store, store.create_job(vid))
    assert job.state == "succeeded"
    assert log[-1] == "shuttle:candidates"
    assert "prefetch" in log and log.index("waited") > log.index("players")
    shuttle = next(s for s in job.stages if s.name == "shuttle")
    players_run = next(s for s in job.stages if s.name == "players")
    assert shuttle.started_at <= players_run.finished_at  # really overlapped


def test_parallel_off_runs_in_order(tmp_path, sample_video, monkeypatch):
    store = Store(tmp_path)
    vid = _video(store, sample_video)
    log = []
    _prefetch_stage(monkeypatch, log)
    called = []
    monkeypatch.setattr(runner, "load_prefetch", lambda n: called.append(n))
    job = runner.run_job(store, store.create_job(vid), {"parallel_stages": "off"})
    assert called == [] and job.state == "failed"  # fake shuttle needs the prefetch


def test_players_failure_cancels_early_shuttle(tmp_path, sample_video, monkeypatch):
    store = Store(tmp_path)
    vid = _video(store, sample_video)
    log = []

    def players(ctx):
        raise RuntimeError("pose crashed")

    _prefetch_stage(monkeypatch, log, players)
    job = runner.run_job(store, store.create_job(vid))
    states = {s.name: s.state for s in job.stages}
    assert job.state == "failed" and states["players"] == "failed"
    assert states["shuttle"] == "skipped" and "waited" in log  # joined, not left running
