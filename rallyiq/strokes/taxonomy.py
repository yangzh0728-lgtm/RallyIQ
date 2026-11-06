"""Stroke-type codes, display names and the mapping from ShuttleSet labels.

The 12 codes are the merged ShuttleSet classes (25-class setting = "unknown" +
12 types x {Top, Bottom}) without the side prefix, so a model trained on
ShuttleSet (e.g. BST) maps onto them one to one.
"""

from __future__ import annotations

from ..contracts import SHOT_TYPES

ZH = {
    "short_serve": "发短球",
    "long_serve": "发高远球",
    "clear": "高远球",
    "smash": "杀球",
    "drop": "吊球",
    "net_shot": "放网",
    "cross_net": "勾对角",
    "push": "推球",
    "net_kill": "扑球",
    "drive": "平抽",
    "lift": "挑球",
    "block": "挡网",
}

EN = {
    "short_serve": "short serve",
    "long_serve": "long serve",
    "clear": "clear",
    "smash": "smash",
    "drop": "drop",
    "net_shot": "net shot",
    "cross_net": "cross-court net shot",
    "push": "push",
    "net_kill": "net kill",
    "drive": "drive",
    "lift": "lift",
    "block": "block",
}

# ShuttleSet (traditional Chinese) label -> code. The first 12 are the merged
# 25-class names; the rest are the extra 35-class names folded into the
# nearest merged class (the same merge ShuttleSet_merged applies).
SHUTTLESET = {
    "放小球": "net_shot",
    "擋小球": "block",
    "殺球": "smash",
    "挑球": "lift",
    "長球": "clear",
    "平球": "drive",
    "切球": "drop",
    "推球": "push",
    "撲球": "net_kill",
    "勾球": "cross_net",
    "發短球": "short_serve",
    "發長球": "long_serve",
    "點扣": "smash",
    "防守回挑": "lift",
    "後場抽平球": "drive",
    "過渡切球": "drop",
    "防守回抽": "drive",
    "小平球": None,  # ShuttleSet moves these to "unknown"
    "未知球種": None,
}

SERVES = ("short_serve", "long_serve")

assert set(ZH) == set(SHOT_TYPES) == set(EN)
assert {v for v in SHUTTLESET.values() if v} == set(SHOT_TYPES)


def from_shuttleset(label: str) -> tuple[str | None, str | None]:
    """``"Top_殺球"`` -> (``"smash"``, ``"far"``). Side is near/far (Bottom = near the camera)."""
    side = None
    if label.startswith("Top_"):
        side, label = "far", label[4:]
    elif label.startswith("Bottom_"):
        side, label = "near", label[7:]
    return SHUTTLESET.get(label.strip().strip("　")), side
