# Data contracts

Single source of truth: `rallyiq/contracts/models.py` (pydantic v2). The JSON Schemas in
`schemas/` and the examples in `examples/` are generated from it by
`python scripts/export_contracts.py`; a test fails if they are stale. Never edit them by hand.

## Conventions

| Item | Rule |
| --- | --- |
| Time | `frame` = 0-based frame index of the original video; `t_ms` = its time in ms of the original video |
| Image coords | px in original resolution (after container rotation), origin top-left |
| Court coords | metres; origin at a court corner; `x` across 0..6.10, `y` along 0..13.40; `y = 0` is the baseline nearer the camera; net at `y = 6.70` |
| Homography | 3×3 row-major, **image px → court metres**; per calibration segment (camera moves start a new segment) |
| Keypoints | COCO-17 order, `[x, y, conf]` px (`COCO17_KEYPOINTS`) |
| Players | `player_id` `P1`/`P2` is identity across side switches; `side` `near`/`far` is where they stand now |
| Unknowns | nullable fields are `null`, never a fake default |
| `status` | `ok` (可用) / `needs_review` (待复核) / `undeterminable` (不可判定) |
| `source` | shuttle: `detected` / `interpolated`; events: `model` / `human`. Raw predictions are never overwritten |

Court geometry helpers (line model, DLT homography fit, projection, reprojection error) are in
`rallyiq/contracts/geometry.py`.

## Pipeline artifacts

| Stage | Module (owner) | Output model | Example |
| --- | --- | --- | --- |
| `probe` | `rallyiq/pipeline/probe.py` | `VideoMeta` | `video_meta.json` |
| `calibration` | `rallyiq/perception/calibration/stage.py` | `CourtCalibration` | `calibration.json` |
| `players` | `rallyiq/perception/players/stage.py` | `PlayerTracks` | `players.json` |
| `shuttle` | `rallyiq/perception/shuttle/stage.py` | `ShuttleTrack` | `shuttle.json` |
| `events` | `rallyiq/events/stage.py` | `EventsResult` (`Rally`, `ShotEvent`) | `events.json` |
| `overlay` | `rallyiq/pipeline/overlay.py` | `OverlayData` (`OverlayFrame`) | `overlay.json` |

Every stage is `run(ctx: StageContext) -> Model`. `ctx` gives `video_path`, `meta`, `work_dir`,
`config`, `report(progress)` and lazily loaded upstream artifacts (`ctx.calibration`,
`ctx.players`, `ctx.shuttle`, `ctx.events`). The runner saves the return value as
`data/videos/<id>/artifacts/<stage>.json`.

Derived (not pipeline stages): `MatchStats` / `CoachReport` / `MatchReport` from `rallyiq/report`
(`match_report.json`). The coach report is cached as `artifacts/coach.json` and carries the
`stats_fingerprint` it was generated from.

Human edits are append-only `Correction`s (`corrections.json`); the API layers them on the raw
`events` artifact to produce the effective events (`events_effective.json`).
