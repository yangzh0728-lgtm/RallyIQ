"""Tunable thresholds for the rule-based event segmentation.

Distances are expressed relative to the on-screen height of a player
("player heights", H) so the same rules work for near and far players and for
any resolution. Times are milliseconds so the rules do not depend on fps.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EventsConfig:
    # --- trajectory cleaning ---
    outlier_jump_rel: float = 0.8  # a lone point this far (in H) from both neighbours is dropped
    min_shuttle_conf: float = 0.2

    # --- rally segmentation ---
    end_gap_ms: float = 1200.0  # shuttle unseen this long -> rally over
    static_ms: float = 600.0  # shuttle (nearly) still this long -> rally over
    # ... or it fell, stayed within 0.1 player heights for at least landed_ms, and the next observation jumps
    # away faster than landed_jump_rel player heights per second (the track moves to another object or the
    # shuttle is picked up and thrown), at least 1 s into the rally. 0 disables. Own footage (24 labelled
    # rallies): 150 ms removes the one merge (0006 244-257 s) and cuts the median end error 0.41 -> 0.30 s,
    # same for 100-250 ms and jump 40-60; no change on either point set. Combined with
    # min_no_hit_rally_ms=2500 it exposes a real 1.3 s rally without a detected hit, which is then dropped.
    landed_ms: float = 0.0
    landed_jump_rel: float = 60.0
    static_speed_rel: float = 0.35  # H per second below which the shuttle counts as still
    min_rally_ms: float = 700.0
    # Join two rallies when the shuttle was unseen between them for at most this long: a tracking
    # dropout mid-rally (the player hides it, a fast flight is missed) longer than end_gap_ms. 0 disables.
    # Own footage 0529 (19 hand-labelled rallies): 800 ms cut split rallies 4 -> 1, but added a false hit on
    # the court24 point set; 1200 ms already joined two real rallies into one. Off until that is solved.
    bridge_gap_ms: float = 0.0
    min_rally_points: int = 8

    # --- hit candidates ---
    window_ms: float = 130.0  # span used for incoming / outgoing velocity
    max_window_ms: float = 450.0  # how far to reach across an occlusion for a velocity
    min_turn_deg: float = 35.0
    full_turn_deg: float = 110.0  # turn angle that saturates the turn score
    speed_jump_ratio: float = 1.8
    min_hit_gap_ms: float = 280.0  # two hits closer than this are one hit
    max_hitter_dist_rel: float = 0.9  # shuttle further than this (in H) from a racket proxy: no hitter
    # What H is in max_hitter_dist_rel. "player": that player's own height, so a large near player reaches far
    # into the image and wins hits of a small far player. "mixed": geometric mean of the player's height and the
    # height of the player nearest the shuttle, so reach shrinks where the shuttle is among small players.
    hitter_scale: str = "player"
    racket_extension: float = 0.7  # racket head ~ wrist + k * (wrist - elbow)
    min_kp_conf: float = 0.3
    pose_search_frames: int = 3  # look this many frames around a hit for a pose
    landing_window_ms: float = 250.0  # direction change this close to a static end is a landing

    # Broadcasts cut to close-ups and replays between rallies; shuttle-like blobs and large people there make
    # false rallies and hits. Frames of camera segments without a usable calibration (not the court view) are
    # ignored, unless calibrated segments cover less than ``court_view_min_share`` of the video (footage without
    # calibration at all is left alone). Needs the calibration (the stage passes it). On the 14 held-out
    # ShuttleSet clips: rules hit F1 56.4 -> 62.2%, rally F1 55.0 -> 82.1%, learned v1e 70.4 -> 71.3%; 6 of 257
    # labelled hits fall inside masked frames. 0.3-0.5 masked 11 of 14 clips (rally F1 73%); 0.2 adds one
    # replay-heavy broadcast calibrated on 26% of its frames.
    mask_off_court_views: bool = True
    court_view_min_share: float = 0.25

    # --- learned hit scorer (learned.py) ---
    # "auto": bundled model (models/hit_scorer.json) if present; "off": hand-weighted rules; or a JSON path.
    # The model's own thresholds (keep / ok / min_candidate) replace the three below unless overridden.
    hit_scorer: str = "auto"
    side_threshold: float = 0.8  # learned side model: confidence needed to overrule the nearest player
    # Frames added to the reported contact frame. The velocity-impulse peak tends to sit a frame or two
    # before the labelled contact; learn_hits.py estimates this from labels and ships it with the model.
    hit_frame_offset: int = 0

    # --- pose-sequence hit recovery (pose_hits.py) ---
    # "off" (default), "auto" (bundled models/pose_hits.json) or a JSON path. Adds the serve and the
    # missed hit between two same-side hits from the players' swing when the shuttle was not seen.
    pose_hits: str = "off"
    # Drop a shuttle-based hit when the bundled pose model (models/pose_hits.json) sees no swing by the
    # hitter (or, hitter unknown, by anyone) within ``swing_veto_window_ms`` of it: the max ``p_swing`` there
    # is below this. Own footage from a low camera behind the court: the shuttle of a neighbouring court or
    # one carried between points turns near a player who is walking, racket down. 0 disables.
    swing_veto: float = 0.0
    swing_veto_window_ms: float = 250.0
    swing_veto_model: str = "auto"  # "auto" = bundled pose model, or a JSON path
    swing_veto_keep_wrist_v: float = 0.0  # ... unless a wrist still moved this fast (body spans/s); 0 = unused
    swing_veto_keep_serve: bool = False  # ... never drop a rally's first hit
    # ... never drop a hit answered by (or answering) the other side within this many ms; 0 = unused
    swing_veto_keep_exchange_ms: float = 0.0
    # Drop a rally whose hits (at most this many; none counts too) are all by one side: nobody answered
    # across the net, so it is a catch, pick-up or toss between points. 0 disables.
    one_sided_rally_max_hits: int = 0
    # Drop a rally with no hit at all that is shorter than this: a pickup, a tap back over the net
    # or a shuttle rolling on the floor between points. Real serve-and-miss points last longer. 0 disables.
    # Own footage 0529 (19 labelled rallies): 2500 ms cut false rallies 16 -> 6 and lost none; the two real
    # rallies without a detected hit lasted 3.3 and 3.4 s. 0906 side view (5 labelled rallies): no change.
    # 0302 / 0310 (other gyms): the 5 rallies it dropped were all checked by eye as false. Own-footage batches
    # set 2500 in their events settings; the code default stays off (broadcast footage is untested).
    min_no_hit_rally_ms: float = 0.0

    # --- decisions ---
    min_candidate_conf: float = 0.3  # below this a candidate is not reported at all
    keep_threshold: float = 0.45  # candidates below this survive only if alternation needs them
    low_track_quality: float = 0.35
    ok_threshold: float = 0.6
    hitter_margin: float = 0.15  # proximity gap below which the hitter is ambiguous
    # Two hits by the same side closer than this cannot both be right (the opponent's hit in between would need
    # two net crossings): one is handed to the other side if a player there is near the shuttle, else the weaker
    # is dropped. 0 disables.
    same_side_min_gap_ms: float = 0.0
    # A hit kept without a known hitter (nobody near the shuttle, or no pose within pose_search_frames) gets
    # the player the alternation calls for (hits._fill_hitters), looking up to hitter_pose_fallback_frames
    # around the contact for a pose. Such hits stay needs_review (reason hitter_from_sequence).
    # 14 held-out ShuttleSet clips, near/far right among correctly timed hits: rules 87.0 -> 92.3%,
    # v1e 89.7 -> 95.4%; hit F1 rules 61.8 -> 65.4%, v1e 70.4 -> 74.6%. Wrist-only proximity (no racket
    # extension) and best-of-a-few-frames proximity were tried and were worse.
    hitter_fill: bool = True
    hitter_pose_fallback_frames: int = 25
    # A player tapping the shuttle to themselves (juggling it on the racket, feeding it up before a serve)
    # makes runs of "hits" by one side. In a real exchange the side alternates, so a hit whose previous AND
    # next hit in the rally are by its own side, both within this gap, is dropped (``self_feed``). Whole runs
    # (``self_feed_run``): also drop the two ends of a run of 3+. 0 disables.
    # DJI_0009, 50 reviewed rule hits: 2000 ms removes 8 of the 26 non-hits left after the floor gate; the 2 real
    # hits it also removes were both marked unsure by the reviewer. 14 ShuttleSet test clips: rules 65.0 -> 64.8%,
    # v2 76.0 -> 76.3%. Whole runs removed 5 real DJI hits and cost v2 0.8 pp, so ``self_feed_run`` stays off.
    self_feed_max_gap_ms: float = 2000.0
    self_feed_run: bool = False

    # --- contact gates (ideas from shuttle-intelligence: only accept a hit with measured contact evidence) ---
    # Shuttle height at the candidate above the hitter's ankles, in that player's heights. Below this, together
    # with a slow incoming shuttle (next field), the shuttle is at the floor: a pickup, a bounce or a shuttle
    # pushed along the floor between rallies, never a stroke. Rules only (``hit_scorer="off"`` or no bundled
    # model): the learned scorer has these facts as features. 0 disables.
    # DJI_0009, 50 reviewed rule hits: removes 6 of 32 non-hits (pickups, floor pushes) and none of the 18
    # real hits; 14 ShuttleSet test clips: rule hit F1 64.5 -> 65.0%.
    floor_contact_h: float = 0.3
    # ... and only when the shuttle arrives slowly (player heights per second over ``window_ms``): a shuttle lying
    # on or rolling along the floor. A real low lift or net return meets a falling, moving shuttle; on the
    # ShuttleSet test clips every true hit below the height limit arrived faster than ~1.4 H/s.
    floor_max_speed_in: float = 1.1
    # Distance from the hitter's closest wrist to the shuttle, in torso lengths (shoulder-to-hip). Beyond this
    # no racket can reach. Real DJI_0009 hits reached up to ~2.1 torso lengths (wrist + racket), so the
    # shuttle-intelligence limit of 0.9 would drop real hits. Even 2.5 cost rule hit F1 on the 14 ShuttleSet
    # test clips (64.5 -> 61.3%; v2 76.0 -> 69.5%), so it stays off. 0 disables (default).
    max_reach_torso: float = 0.0
    # Reject a candidate whose contact point was filled in by the tracker (not measured). Off by default: on the
    # ShuttleSet test clips rule hit F1 64.5 -> 63.0%, and no DJI_0009 non-hit sits on a filled-in point.
    require_observed_contact: bool = False

    occlusion_gap_ms: float = 150.0  # no observation this close around a hit -> occluded
    long_exchange_ms: float = 2600.0  # no hit for this long inside a rally -> maybe missed one
    default_player_height_px: float = 200.0
