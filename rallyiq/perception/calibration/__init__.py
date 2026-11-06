"""Court calibration: court lines -> homography between image pixels and court metres.

Entry points:
    calibrate_frame(image)                  automatic, one frame
    calibrate_from_points(points, ...)      manual fallback (clicked corners)
    CourtTracker / calibrate_video(path)    whole video, with camera-move detection
    stage.run(ctx)                          pipeline stage (imports the pipeline)

OpenCV is imported lazily so that importing the pipeline stage registry
does not require it (demo mode runs without it).
"""

_EXPORTS = {
    "CalibrationMetrics": "calibrate",
    "FrameCalibration": "calibrate",
    "calibrate_frame": "calibrate",
    "calibrate_from_points": "calibrate",
    "CourtTracker": "tracker",
    "Segment": "tracker",
    "TrackerConfig": "tracker",
    "calibrate_video": "tracker",
}


def __getattr__(name):
    if name in _EXPORTS:
        import importlib

        return getattr(importlib.import_module(f".{_EXPORTS[name]}", __name__), name)
    raise AttributeError(name)


__all__ = [
    "CalibrationMetrics",
    "CourtTracker",
    "FrameCalibration",
    "Segment",
    "TrackerConfig",
    "calibrate_frame",
    "calibrate_from_points",
    "calibrate_video",
]
