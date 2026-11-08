"""Optional backend: BST (Badminton Stroke-type Transformer, skeleton + shuttle).

Code: https://github.com/Va6lue/BST-Badminton-Stroke-type-Transformer (MIT).
Weights: published by the authors on Google Drive, trained on ShuttleSet. ShuttleSet's
labels are MIT (CoachAI-Projects) but the footage is BWF broadcast video and the
weights come without a separate licence. Treat them as research / personal use only
until that is cleared, and do not ship them. Nothing here downloads them.

Usage (``ctx.config["strokes"]``)::

    {"classifier": "bst",
     "bst": {"repo": "/path/to/BST-Badminton-Stroke-type-Transformer",
             "weights": "/path/to/bst_cg_ap_..._seq100.pt",
             "model_name": "BST_CG_AP", "pose_style": "JnB_bone", "seq_len": 100}}

Inputs are built exactly like the authors' ``prepare_train_on_shuttleset.py``
("between 2 hits" clips, ``seq_len`` 100):

* joints: per player, relative to the player's box, divided by the box diagonal,
  centre-aligned; missing player = zeros. Player order: Top (far) then Bottom (near).
* pos: feet midpoint in court coordinates normalised to [0, 1] by the court outline,
  with y running from the far baseline (0) to the near baseline (1).
* shuttle: px divided by the video resolution; unseen = (0, 0).
* a frame without both players is a failed frame: players, positions and shuttle all zero, as in the
  authors' preparation (``shuttle_result[failed_ls, :] = 0``).

The 25 logits (``未知球種`` + 12 Top_ + 12 Bottom_) are folded onto our 12 codes using
the hitter's side; probability mass on the other side or on "unknown" lowers the
reported probabilities.

numpy-only input preparation is tested in CI; running the network needs torch and
the authors' repository checkout.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..contracts import COURT_LENGTH_M, COURT_WIDTH_M, NET_Y_M, SHOT_TYPES, EventsResult, PlayerTracks, ShuttleTrack
from .features import StrokeFeatures
from .taxonomy import from_shuttleset

# Merged ShuttleSet classes, in the order of get_merged_stroke_types().
MERGED = ["放小球", "擋小球", "殺球", "挑球", "長球", "平球", "切球", "推球", "撲球", "勾球", "發短球", "發長球"]
CLASSES_25 = ["未知球種"] + [f"Top_{c}" for c in MERGED] + [f"Bottom_{c}" for c in MERGED]
BONE_PAIRS = [
    (0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
]


@dataclass(frozen=True)
class BSTConfig:
    repo: str | None = None
    weights: str | None = None
    model_name: str = "BST_CG_AP"
    pose_style: str = "JnB_bone"  # J_only | JnB_interp | JnB_bone | Jn2B
    seq_len: int = 100
    max_clip_frames: int = 150  # a "between 2 hits" clip longer than this is cut
    device: str | None = None


class BSTUnavailable(RuntimeError):
    pass


# ----------------------------------------------------------------- input preparation (numpy)


def normalize_joints(kps: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    """(J, 2) px -> box-relative / box diagonal, centre aligned (authors' default)."""
    diag = float(np.linalg.norm(bbox[2:] - bbox[:2])) or 1.0
    out = (kps - bbox[:2]) / diag
    centre = ((bbox[:2] + bbox[2:]) / 2 - bbox[:2]) / diag
    return out - centre


def court_pos(court_xy) -> np.ndarray:
    """Our court metres (y = 0 near baseline) -> authors' [0, 1] frame (y = 0 far baseline)."""
    return np.array([court_xy[0] / COURT_WIDTH_M, 1.0 - court_xy[1] / COURT_LENGTH_M], dtype=np.float32)


def make_seq_len_same(target_len: int, joints, pos, shuttle):
    """Port of the authors' ``make_seq_len_same`` (stride subsampling, zero padding)."""
    video_len = len(pos)
    if video_len > target_len:
        need_padding = (video_len % target_len) > (target_len // 2)
        stride = video_len // target_len + int(need_padding)
        joints, pos, shuttle = joints[::stride][:target_len], pos[::stride][:target_len], shuttle[::stride][:target_len]
        new_len = len(pos)
        pad = target_len - new_len if need_padding else 0
    else:
        new_len = video_len
        pad = target_len - video_len
    if pad:
        joints = np.pad(joints, ((0, pad), (0, 0), (0, 0), (0, 0)))
        pos = np.pad(pos, ((0, pad), (0, 0), (0, 0)))
        shuttle = np.pad(shuttle, ((0, pad), (0, 0)))
    return joints, pos, shuttle, new_len


def pose_features(joints: np.ndarray, style: str) -> np.ndarray:
    """(t, m, J, 2) -> (t, m, P, 2) for the pose style the weights were trained with."""
    def bones(j):
        out = []
        for a, b in BONE_PAIRS:
            ja, jb = j[:, :, a], j[:, :, b]
            out.append(np.where((ja != 0) & (jb != 0), jb - ja, 0.0))
        return np.stack(out, axis=-2)

    def interp(j):
        mids = []
        for a, b in BONE_PAIRS:
            ja, jb = j[:, :, a], j[:, :, b]
            mids.append(np.where((ja != 0) & (jb != 0), (ja + jb) / 2, 0.0))
        return np.concatenate((j, np.stack(mids, axis=-2)), axis=-2)

    if style == "J_only":
        return joints
    if style == "JnB_interp":
        return interp(joints)
    if style == "JnB_bone":
        return np.concatenate((joints, bones(joints)), axis=-2)
    if style == "Jn2B":
        return np.concatenate((interp(joints), bones(joints)), axis=-2)
    raise ValueError(f"unknown pose_style {style!r}")


def build_clip(events: EventsResult, shot_id: str, shuttle: ShuttleTrack | None, players: PlayerTracks | None,
               width: int, height: int, cfg: BSTConfig):
    """Inputs for one shot: (pose (s, 2, P, 2), pos (s, 2, 2), shuttle (s, 2), video_len)."""
    shots = sorted(events.shots, key=lambda s: s.frame)
    i = next(k for k, s in enumerate(shots) if s.shot_id == shot_id)
    s = shots[i]
    nxt = shots[i + 1] if i + 1 < len(shots) and shots[i + 1].rally_id == s.rally_id else None
    end = nxt.frame if nxt else s.frame + cfg.max_clip_frames // 2
    end = min(end, s.frame + cfg.max_clip_frames)
    frames = list(range(s.frame, max(end, s.frame + 1)))
    by_frame = {pf.frame: pf for pf in players.frames} if players else {}
    sh = {p.frame: p for p in shuttle.points} if shuttle else {}

    t = len(frames)
    joints = np.zeros((t, 2, 17, 2), dtype=np.float32)
    pos = np.zeros((t, 2, 2), dtype=np.float32)
    shut = np.zeros((t, 2), dtype=np.float32)
    for k, f in enumerate(frames):
        pf = by_frame.get(f)
        filled = [False, False]
        if pf is not None:
            for p in pf.players:
                if p.on_court is False or not p.keypoints or p.foot_court is None:
                    continue
                m = 0 if p.foot_court[1] > NET_Y_M else 1  # Top (far) first
                kp = np.asarray([[x, y] for x, y, _ in p.keypoints], dtype=np.float32)
                joints[k, m] = normalize_joints(kp, np.asarray(p.bbox, dtype=np.float32))
                pos[k, m] = court_pos(p.foot_court)
                filled[m] = True
        if not all(filled):
            # The authors' preparation marks a frame failed unless both players are found, and zeroes the
            # players AND the shuttle of a failed frame (prepare_train_on_shuttleset.py:
            # ``shuttle_result[failed_ls, :] = 0``). Feed the network the same.
            joints[k] = 0.0
            pos[k] = 0.0
            continue
        q = sh.get(f)
        if q is not None:
            shut[k] = (q.x / width, q.y / height)
    joints, pos, shut, n = make_seq_len_same(cfg.seq_len, joints, pos, shut)
    return pose_features(joints, cfg.pose_style).astype(np.float32), pos, shut, n


def fold_probs(probs25, hitter_side: str | None) -> dict[str, float]:
    """25-way softmax -> our 12 codes, keeping only the hitter's side when it is known."""
    out = dict.fromkeys(SHOT_TYPES, 0.0)
    for label, p in zip(CLASSES_25, probs25):
        code, side = from_shuttleset(label)
        if code is None or (hitter_side is not None and side != hitter_side):
            continue
        out[code] += float(p)
    return out


# ------------------------------------------------------------------------- model (torch)


def load_model(cfg: BSTConfig):
    try:
        import torch
    except ImportError as e:  # pragma: no cover - torch is optional
        raise BSTUnavailable("torch is not installed (pip install -e '.[shuttle]')") from e
    if not cfg.repo or not cfg.weights:
        raise BSTUnavailable("set strokes.bst.repo (BST checkout) and strokes.bst.weights")
    root = Path(cfg.repo).expanduser() / "stroke_classification"
    if not (root / "model" / "bst.py").exists() or not Path(cfg.weights).expanduser().exists():
        raise BSTUnavailable(f"BST code or weights not found ({root}, {cfg.weights})")
    sys.path.insert(0, str(root))
    try:
        from model import bst as bst_mod  # type: ignore[import-not-found]
    finally:
        sys.path.remove(str(root))
    n_bones = len(BONE_PAIRS)
    extra = {"J_only": 0, "JnB_interp": 1, "JnB_bone": 1, "Jn2B": 2}[cfg.pose_style]
    net = getattr(bst_mod, cfg.model_name)(
        in_dim=(17 + n_bones * extra) * 2, n_class=len(CLASSES_25), seq_len=cfg.seq_len, depth_tem=2, depth_inter=1,
    )
    device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(Path(cfg.weights).expanduser(), map_location=device)
    net.load_state_dict(state.get("state_dict", state) if isinstance(state, dict) else state)
    return net.to(device).eval(), device


def make_model_fn(cfg: BSTConfig, width: int, height: int):
    """A ``classify.ModelFn`` running BST on every shot."""
    net, device = load_model(cfg)  # raises BSTUnavailable first if torch/code/weights are missing
    import torch

    def run(events: EventsResult, shuttle, players, feats: list[StrokeFeatures]) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        shots = {s.shot_id: s for s in events.shots}
        for f in feats:
            s = shots[f.shot_id]
            pose, pos, shut, n = build_clip(events, f.shot_id, shuttle, players, width, height, cfg)
            t = lambda a: torch.from_numpy(a)[None].to(device)  # noqa: E731
            with torch.no_grad():
                x = t(pose)
                logits = net(x.view(*x.shape[:-2], -1), t(shut), t(pos), torch.tensor([n], device=device))
                p = torch.softmax(logits[0], dim=-1).cpu().numpy()
            side = None
            if s.hitter_court is not None:
                side = "far" if s.hitter_court[1] > NET_Y_M else "near"
            out[f.shot_id] = fold_probs(p, side)
        return out

    return run
