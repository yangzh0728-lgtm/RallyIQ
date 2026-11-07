"""Knowledge-grounded advice: match weaknesses -> coaching notes -> drills to retest.

1. ``detect_focus`` reads ``MatchStats`` and lists the player's problems as *focus*
   items, each with the metric keys that show it, the stroke / aspect tags to look
   up (``knowledge/coaching/taxonomy.json``) and the metric to retest next time.
2. ``retrieve`` scores every coaching note against a focus (stroke and aspect tags,
   the note's own ``related_metrics``) and keeps the best few, mixing coaches.
3. ``knowledge_context`` bundles both for a provider; ``note_card`` is the compact
   view of a note that goes into the prompt.
4. ``ground_sources`` turns the note ids a provider cites back into ``NoteRef``s.
   Links and titles always come from the note files; ids the provider was not
   given are dropped with a warning.

Notes are loaded from ``RALLYIQ_KNOWLEDGE_DIR`` (default ``knowledge/coaching`` at the
repo root). With no notes the coach still works, just without citations.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from ...contracts import Distribution, MatchStats, Metric, NoteRef, ShotMoment
from ..stats import SHOT_TYPE_LABELS, UNCLASSIFIED, index

DEFAULT_KNOWLEDGE_DIR = Path(__file__).resolve().parents[3] / "knowledge" / "coaching"

# ShotEvent.shot_type -> taxonomy stroke tags used by the notes
STROKE_TAGS: dict[str, list[str]] = {
    "short_serve": ["serve_short"], "long_serve": ["serve_flick"], "serve": ["serve_short", "serve_flick"],
    "clear": ["clear", "backhand_clear"], "smash": ["smash", "jump_smash"], "drop": ["drop", "slice"],
    "net_shot": ["net_shot"], "net": ["net_shot"], "cross_net": ["net_shot"], "push": ["push"],
    "net_kill": ["net_kill"], "kill": ["net_kill"], "drive": ["drive"], "lift": ["lift"],
    "block": ["defense", "net_shot"], "defense": ["defense"],
}
# Chinese labels in the stats distributions -> stroke tags
_LABEL_TAGS: dict[str, list[str]] = {}
for _raw, _tags in STROKE_TAGS.items():
    _lst = _LABEL_TAGS.setdefault(SHOT_TYPE_LABELS.get(_raw, _raw), [])
    _lst += [t for t in _tags if t not in _lst]

# Aspects that never answer a match weakness on their own.
_OFF_TOPIC = {"equipment", "trick_shot"}
_LEVELS = ("beginner", "intermediate", "advanced")


# --------------------------------------------------------------------------- notes


def knowledge_dir() -> Path:
    return Path(os.environ.get("RALLYIQ_KNOWLEDGE_DIR") or DEFAULT_KNOWLEDGE_DIR)


@lru_cache(maxsize=4)
def _load(path: str) -> tuple[dict, ...]:
    notes = []
    for f in sorted(Path(path).glob("notes/*/*.json")):
        try:
            n = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(n, dict) and n.get("id") and n.get("url"):
            notes.append(n)
    return tuple(notes)


def load_notes() -> tuple[dict, ...]:
    return _load(str(knowledge_dir()))


# --------------------------------------------------------------------------- focus


@dataclass
class Focus:
    """One weakness worth a drill."""

    id: str
    player_id: str | None
    title: str
    evidence: list[str]
    strokes: list[str] = field(default_factory=list)
    aspects: list[str] = field(default_factory=list)
    retest_metric: str | None = None
    confidence: str = "low"

    def as_dict(self) -> dict:
        return {"focus_id": self.id, "player_id": self.player_id, "title": self.title, "evidence": self.evidence,
                "retest_metric": self.retest_metric, "confidence": self.confidence}


def _usable(e) -> bool:
    if e is None or e.status == "undeterminable":
        return False
    return e.value is not None if isinstance(e, Metric) else e.n > 0


def _weakest(*confs: str) -> str:
    return min(confs, key=["low", "medium", "high"].index)


def _fmt(v: float) -> str:
    return f"{v:.0f}"


def detect_focus(stats: MatchStats) -> list[Focus]:
    """Weaknesses per player, most important first. Only usable evidence counts."""
    idx = index(stats)

    def m(key: str) -> Metric | None:
        e = idx.get(key)
        return e if isinstance(e, Metric) and _usable(e) else None

    def dist(key: str) -> Distribution | None:
        e = idx.get(key)
        return e if isinstance(e, Distribution) and _usable(e) else None

    out: list[Focus] = []
    pids = [p.player_id for p in stats.players]
    for pid in pids:
        # 1. which stroke ended the rallies they lost
        lost = dist(f"{pid}.lost.last_shot_types")
        lost_own = m(f"{pid}.win.lost_own_last_shot_pct")
        typed = {k: v for k, v in (lost.counts.items() if lost else []) if k != UNCLASSIFIED and k in _LABEL_TAGS}
        if typed:
            label, cnt = max(typed.items(), key=lambda kv: kv[1])
            if cnt >= 3 and cnt / lost.n >= 0.3:
                ev = [lost.key] + ([lost_own.key] if lost_own else [])
                out.append(Focus(f"{pid}.lost_stroke", pid, f"{pid} 失分回合的最后一拍多是{label}（{cnt}/{lost.n}）", ev,
                                 strokes=_LABEL_TAGS[label], aspects=["technique"],
                                 retest_metric=f"{pid}.win.lost_own_last_shot_pct", confidence=lost.confidence))
        if lost_own is not None and lost_own.value >= 55 and not any(f.id == f"{pid}.lost_stroke" for f in out):
            out.append(Focus(f"{pid}.own_errors", pid, f"{pid} 失分多在自己最后一拍（{_fmt(lost_own.value)}%）",
                             [lost_own.key], aspects=["stability", "technique"],
                             retest_metric=lost_own.key, confidence=lost_own.confidence))

        # 2. serve / receive
        serve, recv = m(f"{pid}.win.serve_rate_pct"), m(f"{pid}.win.receive_rate_pct")
        if serve is not None and serve.value < 40:
            out.append(Focus(f"{pid}.serve", pid, f"{pid} 发球回合胜率偏低（{_fmt(serve.value)}%）", [serve.key],
                             strokes=["serve_short", "serve_flick"], aspects=["tactics"],
                             retest_metric=serve.key, confidence=serve.confidence))
        if recv is not None and recv.value < 40:
            out.append(Focus(f"{pid}.receive", pid, f"{pid} 接发球回合胜率偏低（{_fmt(recv.value)}%）", [recv.key],
                             strokes=["push", "net_kill", "net_shot", "lift"], aspects=["split_step", "tactics"],
                             retest_metric=recv.key, confidence=recv.confidence))

        # 3. long vs short rallies
        short, long_ = m(f"{pid}.win.short_rate_pct"), m(f"{pid}.win.long_rate_pct")
        if short is not None and long_ is not None and abs(short.value - long_.value) >= 20:
            conf = _weakest(short.confidence, long_.confidence)
            if long_.value < short.value:
                out.append(Focus(f"{pid}.long_rallies", pid,
                                 f"{pid} 长回合胜率（{_fmt(long_.value)}%）低于短回合（{_fmt(short.value)}%）",
                                 [long_.key, short.key], strokes=["defense", "drive", "clear"],
                                 aspects=["stability", "physical", "tactics"], retest_metric=long_.key, confidence=conf))
            else:
                out.append(Focus(f"{pid}.short_rallies", pid,
                                 f"{pid} 短回合胜率（{_fmt(short.value)}%）低于长回合（{_fmt(long_.value)}%）",
                                 [short.key, long_.key], strokes=["serve_short", "net_shot", "net_kill", "push"],
                                 aspects=["tactics", "split_step"], retest_metric=short.key, confidence=conf))

        # 4. movement and court coverage
        cover = m(f"{pid}.cover.ratio_pct")
        if cover is not None and cover.value < 45:
            out.append(Focus(f"{pid}.coverage", pid, f"{pid} 场地覆盖偏窄（{_fmt(cover.value)}%）", [cover.key],
                             aspects=["footwork", "split_step"], retest_metric=cover.key, confidence=cover.confidence))
        per_rally = m(f"{pid}.move.per_rally_m")
        others = [m(f"{o}.move.per_rally_m") for o in pids if o != pid]
        others = [o for o in others if o is not None and o.value > 0]
        if per_rally is not None and others and len(pids) == 2 and per_rally.value / others[0].value >= 1.25:
            o = others[0]
            out.append(Focus(f"{pid}.run_more", pid,
                             f"{pid} 每回合跑动 {per_rally.value:.1f} 米，对手 {o.value:.1f} 米，可能常被调动",
                             [per_rally.key, o.key], aspects=["footwork", "split_step", "tactics"],
                             retest_metric=per_rally.key, confidence=_weakest(per_rally.confidence, o.confidence)))
        depth = dist(f"{pid}.pos.depth_zones")
        if depth is not None:
            zone, cnt = max(depth.counts.items(), key=lambda kv: kv[1])
            share = 100 * cnt / depth.n
            if share >= 50 and zone in ("后场", "前场"):
                strokes = ["net_shot", "net_kill", "lift"] if zone == "后场" else ["clear", "defense", "smash"]
                out.append(Focus(f"{pid}.depth", pid, f"{pid} 回合中约 {_fmt(share)}% 的时间停在{zone}", [depth.key],
                                 strokes=strokes, aspects=["footwork"], retest_metric=f"{pid}.pos.depth_mean_m",
                                 confidence=depth.confidence))
    return out


# --------------------------------------------------------------------------- retrieval


def _metric_templates(f: Focus) -> set[str]:
    keys = set(f.evidence) | ({f.retest_metric} if f.retest_metric else set())
    out = set()
    for k in keys:
        if f.player_id and k.startswith(f.player_id + "."):
            out.add("{pid}." + k[len(f.player_id) + 1:])
        else:
            out.add(k)
    return out


def score_note(note: dict, f: Focus, level: str | None = None) -> float:
    strokes = set(note.get("strokes") or [])
    aspects = set(note.get("aspects") or [])
    s_hit = len(strokes & set(f.strokes))
    a_hit = len(aspects & set(f.aspects))
    if f.strokes and not s_hit and not (a_hit and not strokes):
        # a stroke problem needs a note about that stroke (or a general one on the same aspect)
        return 0.0
    if not s_hit and not a_hit:
        return 0.0
    if aspects and aspects <= _OFF_TOPIC or strokes and strokes <= _OFF_TOPIC:
        return 0.0
    score = 3.0 * s_hit + 2.0 * a_hit
    score += 2.0 * len(set(note.get("related_metrics") or []) & _metric_templates(f))
    score += min(len(note.get("drills") or []), 3) * 0.3 + min(len(note.get("corrections") or []), 2) * 0.2
    if strokes and f.strokes:
        score -= 0.4 * len(strokes - set(f.strokes))  # prefer notes focused on this stroke
    if level and note.get("level") not in (level, "all"):
        score -= 1.0
    return score


def retrieve(f: Focus, notes: tuple[dict, ...] | list[dict], k: int = 3, level: str | None = None) -> list[dict]:
    scored = sorted(((score_note(n, f, level), n) for n in notes), key=lambda sn: (-sn[0], sn[1]["id"]))
    scored = [(s, n) for s, n in scored if s > 0]
    picked: list[dict] = []
    coaches: set[str] = set()
    # first pass: best note per coach, so advice is not all from one source
    for _, n in scored:
        if len(picked) < k and n.get("coach") not in coaches:
            picked.append(n)
            coaches.add(n.get("coach"))
    for _, n in scored:
        if len(picked) >= k:
            break
        if n not in picked:
            picked.append(n)
    return picked


def _trim(xs, n: int) -> list:
    return list(xs or [])[:n]


def note_card(note: dict) -> dict:
    """What the model sees of a note: enough to coach from, small enough for ~20 notes."""
    return {
        "note_id": note["id"], "coach": note.get("coach"), "title": note.get("title"), "level": note.get("level"),
        "topic": note.get("topic"), "strokes": note.get("strokes"), "aspects": note.get("aspects"),
        "key_points": _trim(note.get("key_points"), 6), "common_mistakes": _trim(note.get("common_mistakes"), 4),
        "corrections": _trim(note.get("corrections"), 3), "drills": _trim(note.get("drills"), 4),
        "coach_cues": _trim(note.get("coach_cues"), 6), "self_check": _trim(note.get("self_check"), 3),
        "when_to_use": _trim(note.get("when_to_use"), 2), "related_metrics": note.get("related_metrics") or [],
    }


@dataclass
class Knowledge:
    focus: list[Focus]
    notes: dict[str, dict]  # note id -> note, every note retrieved for any focus
    by_focus: dict[str, list[str]]  # focus id -> note ids, best first
    moments: list[ShotMoment] = field(default_factory=list)  # shots to change (the user's when known)
    self_player: str | None = None  # the user (我), when known

    def prompt_payload(self) -> list[dict]:
        return [dict(f.as_dict(), notes=[note_card(self.notes[i]) for i in self.by_focus.get(f.id, [])])
                for f in self.focus]


def select_moments(stats: MatchStats, self_player: str | None = None) -> list[ShotMoment]:
    """The user's shots to change when we know who the user is, else everyone's."""
    if self_player and any(m.player_id == self_player for m in stats.moments):
        return [m for m in stats.moments if m.player_id == self_player]
    if self_player and stats.moments and self_player in {p.player_id for p in stats.players}:
        return []  # the user has no flagged shot: do not advise on the opponent's
    return list(stats.moments)


def knowledge_context(stats: MatchStats, notes: tuple[dict, ...] | list[dict] | None = None, k: int = 3,
                      max_focus: int = 6, level: str | None = None, self_player: str | None = None) -> Knowledge:
    notes = load_notes() if notes is None else notes
    level = level or os.environ.get("RALLYIQ_PLAYER_LEVEL") or None
    if level not in _LEVELS:
        level = None
    focus = detect_focus(stats)[:max_focus]
    by_focus: dict[str, list[str]] = {}
    found: dict[str, dict] = {}
    for f in focus:
        got = retrieve(f, notes, k=k, level=level)
        by_focus[f.id] = [n["id"] for n in got]
        found.update({n["id"]: n for n in got})
    return Knowledge(focus=focus, notes=found, by_focus=by_focus, moments=select_moments(stats, self_player),
                     self_player=self_player)


# --------------------------------------------------------------------------- citations


def _drill_t(note: dict, drill: str | None) -> int | None:
    for d in note.get("drills") or []:
        if drill and d.get("name") == drill and isinstance(d.get("t_s"), int | float):
            return int(d["t_s"])
    return None


def ref(note: dict, drill: str | None = None, t_s: int | None = None) -> NoteRef:
    drills = {d.get("name") for d in note.get("drills") or []}
    drill = drill if drill in drills else None
    t = t_s if t_s is not None else _drill_t(note, drill)
    return NoteRef(note_id=note["id"], coach=str(note.get("coach", "")), title=str(note.get("title", "")),
                   url=str(note["url"]), t_s=t, drill=drill)


def ground_sources(raw: list, knowledge: Knowledge | None) -> tuple[list[NoteRef], list[str]]:
    """Provider citations -> NoteRefs. Accepts ``"note_id"`` or ``{"note_id", "drill"}`` items."""
    refs: list[NoteRef] = []
    unknown: list[str] = []
    seen: set[str] = set()
    for item in raw or []:
        nid, drill = (item, None) if isinstance(item, str) else (item.get("note_id"), item.get("drill"))
        if not nid or nid in seen:
            continue
        note = knowledge.notes.get(nid) if knowledge else None
        if note is None:
            unknown.append(str(nid))
            continue
        seen.add(nid)
        refs.append(ref(note, drill))
    warnings = [f"引用了未提供的教学笔记：{', '.join(unknown)}"] if unknown else []
    return refs, warnings


def video_link(r: NoteRef) -> str:
    """Source URL, opened at ``t_s`` when known (YouTube ``t=``, Bilibili ``t=``)."""
    if r.t_s is None:
        return r.url
    sep = "&" if "?" in r.url else "?"
    return f"{r.url}{sep}t={r.t_s}"
