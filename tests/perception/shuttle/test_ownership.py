"""A next-court rally in the same pixels as our air space (train_v5: sam's S118 / S119)."""

import math

from rallyiq.contracts.models import PlayerDetection, PlayerFrame, PlayerTracks
from rallyiq.perception.shuttle.ownership import CourtOwnership
from rallyiq.perception.shuttle.tracker import ShuttleTracker, TrackerConfig

FPS, W, H = 30.0, 1920, 1080


def person(pid, side, x, y1, y2, wrist):
    kps = [[0.0, 0.0, 0.0] for _ in range(17)]
    kps[9] = [wrist[0], wrist[1], 0.9]
    kps[10] = [wrist[0], wrist[1], 0.9]
    return PlayerDetection(player_id=pid, track_id=0, team="A" if pid == "P1" else "B", side=side,
                           bbox=[x - 60, y1, x + 60, y2], keypoints=kps, foot_px=[x, y2], foot_court=[3.0, 3.0],
                           on_court=True, confidence=0.9)


def players(n, hits):
    """Near player at x=700, far at x=1400; a wrist swings fast around each hit frame."""
    frames = []
    for f in range(n):
        out = []
        for pid, side, x, y1, y2 in (("P2", "near", 700, 400, 850), ("P1", "far", 1400, 550, 750)):
            h = y2 - y1
            d = min((abs(f - t) for t, p in hits if p == pid), default=99)
            off = h * 0.6 * (d if d < 4 else 4) / 4
            out.append(person(pid, side, x, y1, y2, (x + off, y1 + off * 0.2)))
        frames.append(PlayerFrame(frame=f, t_ms=int(f * 1000 / FPS), players=out))
    return PlayerTracks(video_id="v", model="m", match_format="singles", frames=frames)


def arc(f0, f1, a, b, lift, score):
    out = {}
    for f in range(f0, f1 + 1):
        u = (f - f0) / (f1 - f0)
        out[f] = (a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u - lift * 4 * u * (1 - u), score)
    return out


def scene(n=240):
    """Our rally between the players, and a longer, higher-scoring one out to the left."""
    hits, ours, theirs = [], {}, {}
    near, far = (700.0, 400.0), (1400.0, 550.0)
    f, side = 5, 0
    while f + 30 < n:
        a, b = (near, far) if side == 0 else (far, near)
        hits.append((f, "P2" if side == 0 else "P1"))
        ours.update(arc(f, f + 28, a, b, 200, 0.6))
        f, side = f + 30, 1 - side
    hits.append((f, "P2" if side == 0 else "P1"))
    f, side = 0, 0
    while f + 24 < n:
        a, b = ((60.0, 420.0), (430.0, 430.0)) if side == 0 else ((430.0, 430.0), (60.0, 420.0))
        theirs.update(arc(f, f + 24, a, b, 120, 0.8))
        f, side = f + 24, 1 - side
    cands = {f: [c for c in (theirs.get(f), ours.get(f)) if c] for f in range(n)}
    return cands, players(n, hits), ours, theirs


def run(cands, pl, n, own=True):
    o = CourtOwnership(pl, FPS, W)
    t = ShuttleTracker(FPS, W, H, TrackerConfig(hot_mode="off"), foreign=o.logit if own else None,
                       contact=o.contact if own else None)
    res = t.track(cands, n)
    return t, {p.frame: (p.x, p.y) for p in res.points}


def share(points, ref):
    return sum(1 for f, (x, y, _s) in ref.items() if f in points and math.dist(points[f], (x, y)) < 5) / len(ref)


def test_neighbour_rally_is_rejected():
    cands, pl, ours, theirs = scene()
    _, before = run(cands, pl, 240, own=False)
    assert share(before, theirs) > 0.9  # longer and stronger: it used to win
    t, after = run(cands, pl, 240)
    assert t.concurrency > 0.5
    assert share(after, theirs) < 0.05
    assert share(after, ours) > 0.9


def test_single_court_is_left_alone():
    cands, pl, ours, theirs = scene()
    only = {f: [c for c in cs if c == theirs.get(f)] for f, cs in cands.items()}
    t, pts = run(only, pl, 240)
    assert t.concurrency == 0.0
    assert share(pts, theirs) > 0.9  # nobody else in view: no evidence of another court, keep it


def test_unknown_without_both_halves():
    _, pl, _, theirs = scene()
    for fr in pl.frames:
        fr.players = [p for p in fr.players if p.side == "near"]
    o = CourtOwnership(pl, FPS, W)
    fs = sorted(theirs)[:20]
    assert o.logit(fs, [theirs[f][:2] for f in fs], []) is None


def test_clean_flight_takes_its_frames_back():
    """S119: a tracklet jumping about at lower scores must not outweigh a hit-to-hit flight."""
    pl = players(60, [(10, "P2"), (30, "P1")])
    flight = arc(10, 30, (720.0, 400.0), (1380.0, 555.0), 200, 0.7)
    blob = {f: (900.0 + 40 * math.sin(f), 520.0 + 30 * math.cos(1.7 * f), 0.5) for f in range(5, 58)}
    cands = {f: [c for c in (blob.get(f), flight.get(f)) if c] for f in range(60)}
    o = CourtOwnership(pl, FPS, W)
    t = ShuttleTracker(FPS, W, H, TrackerConfig(hot_mode="off"), contact=o.contact)
    res = t.track(cands, 60)
    pts = {p.frame: (p.x, p.y) for p in res.points}
    assert share(pts, flight) > 0.9
