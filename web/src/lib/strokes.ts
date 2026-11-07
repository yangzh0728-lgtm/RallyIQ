import type { EventsResult, PlayerId, ShotEvent, ShotType, StrokeAttr, StrokeDetail, StrokeInfo, StrokesResult } from "./types";

// Display names for stroke types. Mirrors rallyiq/strokes/taxonomy.py.

export const SHOT_TYPES: ShotType[] = [
  "short_serve", "long_serve", "clear", "smash", "drop", "net_shot", "cross_net", "push", "net_kill", "drive", "lift", "block",
];

export const SHOT_TYPE_ZH: Record<ShotType, string> = {
  short_serve: "发短球",
  long_serve: "发高远球",
  clear: "高远球",
  smash: "杀球",
  drop: "吊球",
  net_shot: "放网",
  cross_net: "勾对角",
  push: "推球",
  net_kill: "扑球",
  drive: "平抽",
  lift: "挑球",
  block: "挡网",
};

/** Hue per type family: attacking (red), overhead (amber), net (green), flat (blue), defence/lift (violet), serve (grey). */
export const SHOT_TYPE_COLOR: Record<ShotType, string> = {
  short_serve: "#94a3b8",
  long_serve: "#cbd5e1",
  clear: "#fbbf24",
  smash: "#f87171",
  drop: "#fb923c",
  net_shot: "#34d399",
  cross_net: "#6ee7b7",
  push: "#60a5fa",
  net_kill: "#ef4444",
  drive: "#38bdf8",
  lift: "#a78bfa",
  block: "#c4b5fd",
};

/** Review reason codes from rallyiq/strokes (features.py, classify.py). */
const REASON_ZH: Record<string, string> = {
  hitter_unknown: "击球者未知",
  no_hitter_pose: "击球者骨架缺失",
  no_shuttle_at_hit: "击球帧看不到球",
  no_court_position: "无球场坐标（标定未成功）",
  last_shot_of_rally: "回合最后一拍（无接球方）",
  next_hit_same_side: "下一拍在同一半场（可能漏检）",
  flight_not_observed: "飞行段几乎没有观测",
  first_hit_not_serve_like: "首拍不像发球（发球可能漏检）",
  ambiguous_types: "前两种类型接近",
  hit_needs_review: "击球本身待复核",
  incomplete_evidence: "证据不完整",
  no_rule_fits: "没有规则匹配",
  insufficient_evidence: "证据不足",
};

export const reasonLabel = (code: string): string => REASON_ZH[code] ?? code;
export const typeLabel = (t: ShotType | null | undefined): string => (t ? SHOT_TYPE_ZH[t] ?? t : "未判定");

const EVIDENCE_ZH: Record<string, [string, (v: number) => string]> = {
  contact_h: ["击球点高度", (v) => `${v.toFixed(2)} 身高（1=鼻）`],
  hitter_depth_m: ["击球者离网", (v) => `${v.toFixed(1)} m`],
  receiver_depth_m: ["接球点离网", (v) => `${v.toFixed(1)} m`],
  lateral_m: ["横向距离", (v) => `${v.toFixed(1)} m`],
  flight_ms: ["飞行时间", (v) => `${Math.round(v)} ms`],
  court_speed_mps: ["平均水平速度", (v) => `${v.toFixed(1)} m/s`],
  rise: ["弧线高度", (v) => `${v.toFixed(2)} 身高`],
  left_frame: ["飞出画面", (v) => (v ? "是" : "否")],
};

export function evidenceRows(e: Record<string, number | null> | undefined): [string, string][] {
  if (!e) return [];
  return Object.entries(EVIDENCE_ZH).map(([k, [label, fmt]]) => {
    const v = e[k];
    return [label, v === null || v === undefined ? "—" : fmt(v)];
  });
}

/** Attach stage-`strokes` predictions to raw events. Port of rallyiq/strokes/classify.py:merge_strokes. */
export function mergeStrokes(events: EventsResult, strokes: StrokesResult | null | undefined): EventsResult {
  if (!strokes) return events;
  const byId = new Map<string, StrokeInfo>(strokes.shots.map((s) => [s.shot_id, s.stroke]));
  return {
    ...events,
    shots: events.shots.map((s) => {
      const st = byId.get(s.shot_id);
      return st ? { ...s, stroke: st, shot_type: st.shot_type } : s;
    }),
  };
}

export interface StrokeStats {
  player: PlayerId;
  total: number; // shots by this player
  typed: number; // with a stroke type
  review: number; // typed but needs review (model)
  counts: { type: ShotType; n: number; review: number }[]; // most frequent first
}

/** Per-player stroke-type distribution. Undetermined shots are counted in total only. */
export function strokeStats(shots: ShotEvent[], players: PlayerId[]): StrokeStats[] {
  return players.map((player) => {
    const own = shots.filter((s) => s.hitter === player);
    const counts = new Map<ShotType, { n: number; review: number }>();
    let review = 0;
    for (const s of own) {
      const t = s.shot_type;
      if (!t) continue;
      const c = counts.get(t) ?? { n: 0, review: 0 };
      c.n += 1;
      if (s.stroke && s.stroke.source === "model" && s.stroke.status !== "ok") {
        c.review += 1;
        review += 1;
      }
      counts.set(t, c);
    }
    const list = [...counts.entries()].map(([type, c]) => ({ type, ...c })).sort((a, b) => b.n - a.n || SHOT_TYPES.indexOf(a.type) - SHOT_TYPES.indexOf(b.type));
    return { player, total: own.length, typed: list.reduce((a, c) => a + c.n, 0), review, counts: list };
  });
}

/** Display names for the layers of StrokeDetail. Mirrors rallyiq/strokes/detail.py. */
export const DETAIL_ZH: Record<string, string> = {
  forehand: "正手",
  backhand: "反手",
  around_head: "头顶",
  straight: "直线",
  cross: "斜线",
  high: "击球点高",
  low: "击球点低",
  fast: "快",
  medium: "中速",
  slow: "慢",
  attacking: "主动进攻",
  neutral: "相持",
  defensive: "被动防守",
  attack: "进攻",
  move: "调动",
  control: "控制",
  defend: "防守",
  transition: "过渡",
};

export const DETAIL_SOURCE_ZH: Record<StrokeAttr["source"], string> = {
  model: "模型识别",
  measured: "按轨迹测量",
  rule: "按规则推断，仅供参考",
};

/** The attribute chips of a stroke, in display order (empty when the stroke has no detail). */
export function detailChips(stroke: StrokeInfo | null | undefined): { key: string; label: string; title: string; inferred: boolean }[] {
  const d = stroke?.detail;
  if (!d) return [];
  const order: [string, StrokeAttr | null | undefined][] = [
    ["hand", d.hand], ["direction", d.direction], ["contact", d.contact], ["pace", d.pace], ["role", d.role], ["intent", d.intent],
  ];
  return order
    .filter(([, a]) => a && DETAIL_ZH[a.value])
    .map(([key, a]) => ({
      key,
      label: DETAIL_ZH[a!.value],
      title: `${DETAIL_SOURCE_ZH[a!.source]}${a!.p != null ? `（把握 ${Math.round(a!.p * 100)}%）` : ""}`,
      inferred: a!.source === "rule",
    }));
}

// ---- mirror of rallyiq/strokes/detail.py retype(): a corrected stroke type re-derives the rule layers.
const FAMILY: Partial<Record<ShotType, string>> = { short_serve: "serve", long_serve: "serve", cross_net: "net_shot" };
const SUBTYPE: Partial<Record<ShotType, string>> = { cross_net: "cross_court_net_shot", short_serve: "short_serve", long_serve: "long_serve" };
const DESC_ZH: Record<string, string> = {
  forehand: "正手", backhand: "反手", around_head: "头顶", straight: "直线", cross: "斜线",
  high: "击球点较高", low: "击球点较低", fast: "出球快", medium: "", slow: "出球慢",
  attacking: "主动进攻", neutral: "相持", defensive: "被动防守",
};
const INTENT_ZH: Record<string, string> = {
  attack: "直接施压，逼对手被动或失误",
  move: "调动对手前后跑动，拉开空当",
  control: "控制网前，逼对手起球，为下一拍进攻创造机会",
  defend: "化解对手进攻，先保证回球质量",
  transition: "争取时间回位，由被动转入相持",
};

function roleOf(t: ShotType, contact: string | null): string {
  const fam = FAMILY[t] ?? t;
  if (fam === "smash" || fam === "net_kill") return "attacking";
  if (fam === "push" || fam === "drive" || fam === "drop") return contact === "high" ? "attacking" : "neutral";
  if (fam === "clear") return contact === "low" ? "defensive" : "neutral";
  if (fam === "block" || fam === "lift") return contact === "low" ? "defensive" : "neutral";
  return "neutral";
}

function intentOf(t: ShotType, role: string): string {
  const fam = FAMILY[t] ?? t;
  if (fam === "smash" || fam === "net_kill" || ((fam === "push" || fam === "drive") && role === "attacking")) return "attack";
  if (((fam === "drop" || fam === "clear") && role !== "defensive") || t === "long_serve") return "move";
  if (fam === "net_shot" || fam === "serve" || fam === "push" || fam === "drive") return "control";
  if (fam === "block" || (fam === "lift" && role === "defensive")) return "defend";
  return "transition";
}

function describe(t: ShotType, d: StrokeDetail): string {
  const hand = d.hand && FAMILY[t] !== "serve" ? DESC_ZH[d.hand.value] : "";
  const dir = d.direction && t !== "cross_net" && FAMILY[t] !== "serve" ? DESC_ZH[d.direction.value] : "";
  const parts = [hand + dir + SHOT_TYPE_ZH[t]];
  if (d.contact) parts.push(DESC_ZH[d.contact.value]);
  if (d.pace && DESC_ZH[d.pace.value]) parts.push(DESC_ZH[d.pace.value]);
  if (d.role) parts.push("属于" + DESC_ZH[d.role.value]);
  if (d.intent) parts.push("意图是" + INTENT_ZH[d.intent.value]);
  return parts.join("，") + "。";
}

export function retypeDetail(d: StrokeDetail | null | undefined, t: ShotType | null): StrokeDetail | null {
  if (!d || !t) return null;
  const fam = FAMILY[t] ?? t;
  const role = roleOf(t, d.contact?.value ?? null);
  const out: StrokeDetail = {
    ...d,
    family: fam,
    subtype: SUBTYPE[t] ?? null,
    hand: fam === "serve" ? null : d.hand ?? null,
    direction: fam === "serve" ? null : d.direction ?? null,
    pace: fam === d.family ? d.pace ?? null : null,
    role: { value: role, p: null, source: "rule" },
    intent: { value: intentOf(t, role), p: null, source: "rule" },
  };
  return { ...out, description_zh: describe(t, out) };
}
