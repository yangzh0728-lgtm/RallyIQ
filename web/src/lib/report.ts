import type {
  CoachInsight,
  CoachReport,
  Distribution,
  EventsResult,
  InsightKind,
  MatchStats,
  Metric,
  MetricConfidence,
  NoteRef,
  OverlayData,
  PlayerId,
  PlayerStats,
  Status,
} from "./types";

// ---------------------------------------------------------------- display helpers

/** Source video, opened at ``t_s`` when the note knows the moment. */
export function videoLink(r: NoteRef): string {
  if (r.t_s == null) return r.url;
  return `${r.url}${r.url.includes("?") ? "&" : "?"}t=${r.t_s}`;
}

export const STATUS_LABEL: Record<Status, string> = { ok: "可用", needs_review: "待复核", undeterminable: "不可判定" };
export const CONF_LABEL: Record<MetricConfidence, string> = { high: "可信度高", medium: "可信度中", low: "可信度低" };
export const KIND_LABEL: Record<InsightKind, string> = {
  strength: "优势",
  weakness: "问题",
  pattern: "规律",
  tactic: "战术",
  data_quality: "数据质量",
};

export function fmtValue(m: Pick<Metric, "value" | "unit">): string {
  if (m.value === null) return "—";
  const v = Math.abs(m.value - Math.round(m.value)) < 0.05 ? m.value.toFixed(0) : m.value.toFixed(1);
  if (!m.unit) return v;
  return m.unit === "%" ? `${v}%` : `${v} ${m.unit}`;
}

export type Evidence = Metric | Distribution;

export function isMetric(e: Evidence): e is Metric {
  return "value" in e;
}

/** All metrics and distributions by key. */
export function indexStats(s: MatchStats): Map<string, Evidence> {
  const out = new Map<string, Evidence>();
  const add = (xs: Evidence[]) => xs.forEach((x) => out.set(x.key, x));
  add(s.metrics);
  add(s.distributions);
  s.players.forEach((p) => {
    add(p.metrics);
    add(p.distributions);
  });
  return out;
}

export function evidenceText(e: Evidence): string {
  if (isMetric(e)) return `${e.label}：${fmtValue(e)}（n=${e.n}，${STATUS_LABEL[e.status]}）`;
  const parts = Object.entries(e.counts).map(([k, c]) => `${k} ${c}`);
  return `${e.label}：${parts.join("，") || "无数据"}（n=${e.n}，${STATUS_LABEL[e.status]}）`;
}

/** Client-side Markdown export (mock mode); the backend renders its own. */
export function reportMarkdown(title: string, stats: MatchStats, coach: CoachReport | null, stale: boolean): string {
  const out: string[] = [`# ${title}`, ""];
  if (coach) {
    out.push("## AI 教练点评", "");
    if (stale) out.push("> 注意：这份点评生成后数据又有修正，内容可能已过时。", "");
    out.push(coach.summary, "");
    coach.insights.forEach((i) => {
      out.push(`- **[${KIND_LABEL[i.kind]}] ${i.title}**（${CONF_LABEL[i.confidence]}${i.grounded ? "" : "，含未核实数字"}）  `, `  ${i.detail}`);
    });
    if (coach.shot_advice?.length) {
      out.push("", "### 逐拍建议", "");
      coach.shot_advice.forEach((a) => {
        out.push(`- **${a.timestamp}** ${a.situation}（${CONF_LABEL[a.confidence]}）  `, `  ${a.advice}  `, `  ${a.why}`);
      });
    }
    if (coach.training.length) {
      out.push("", "### 训练建议", "");
      coach.training.forEach((t) => {
        out.push(`1. ${t.player_id ? `${t.player_id}：` : ""}**${t.drill}** — ${t.focus ? `针对：${t.focus}。` : ""}${t.why}`);
        if (t.key_points?.length) out.push(`   要点：${t.key_points.join("；")}`);
        for (const r of t.sources ?? []) out.push(`   参考：${r.coach}《${r.title}》${r.drill ? ` · 练习「${r.drill}」` : ""} ${videoLink(r)}`);
      });
    }
    out.push("");
  }
  const caveats = coach?.caveats ?? stats.caveats;
  if (caveats.length) out.push("## 数据说明", "", ...caveats.map((c) => `- ${c}`), "");
  const table = (ms: Metric[]) => [
    "| 指标 | 数值 | 样本 | 状态 |",
    "| --- | --- | --- | --- |",
    ...ms.map((m) => `| ${m.label} | ${fmtValue(m)} | ${m.n} | ${STATUS_LABEL[m.status]} |`),
  ];
  out.push("## 全场统计", "", ...table(stats.metrics), "");
  stats.players.forEach((p) => out.push(`## ${p.player_id}`, "", ...table(p.metrics), ""));
  return out.join("\n") + "\n";
}

// ---------------------------------------------------------------- mock (?mock): approximate stats in the browser

const MOCK_NOTE = "演示数据：前端近似计算";

function level(n: number, okN: number, reviewN: number): { status: Status; confidence: MetricConfidence } {
  if (n >= okN) return { status: "ok", confidence: "high" };
  if (n >= reviewN) return { status: "needs_review", confidence: "medium" };
  if (n > 0) return { status: "needs_review", confidence: "low" };
  return { status: "undeterminable", confidence: "low" };
}

function metric(key: string, label: string, value: number | null, unit: string | null, n: number, okN: number, reviewN: number, note: string | null = null): Metric {
  const l = level(value === null ? 0 : n, okN, reviewN);
  return { key, label, value: value === null ? null : Math.round(value * 10) / 10, unit, n, ...l, note };
}

function dist(key: string, label: string, counts: Record<string, number>, okN: number, reviewN: number, note: string | null = null): Distribution {
  const n = Object.values(counts).reduce((a, b) => a + b, 0);
  return { key, label, counts, n, ...level(n, okN, reviewN), note };
}

const pct = (a: number, b: number) => (b ? (100 * a) / b : null);

function hashStr(s: string): string {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 16777619);
  return (h >>> 0).toString(16).padStart(8, "0");
}

export function mockStats(ev: EventsResult, overlay: OverlayData): MatchStats {
  const rallies = [...ev.rallies].sort((a, b) => a.start_frame - b.start_frame);
  const lens = rallies.map((r) => r.shot_ids.length);
  const durs = rallies.map((r) => (r.end_ms - r.start_ms) / 1000);
  const nr = rallies.length;
  const mean = (xs: number[]) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null);
  const decided = rallies.filter((r) => r.winner);
  const confirmed = ev.shots.filter((s) => s.status === "ok" || s.source === "human").length;

  const players: PlayerStats[] = (["P1", "P2"] as PlayerId[]).map((pid) => {
    const shots = ev.shots.filter((s) => s.hitter === pid);
    const known = ev.shots.filter((s) => s.hitter).length;
    // movement: foot positions inside rallies, glitch steps (> 8 m/s) dropped
    let dist_m = 0;
    let samples = 0;
    const depth: Record<string, number> = { 后场: 0, 中场: 0, 前场: 0 };
    for (const r of rallies) {
      let prev: [number, number, number] | null = null;
      for (const f of overlay.frames) {
        if (f.frame < r.start_frame || f.frame > r.end_frame) continue;
        const p = f.players.find((x) => x.player_id === pid && x.foot_court && !x.interpolated);
        if (!p?.foot_court) continue;
        const [x, y] = p.foot_court;
        samples++;
        const d = y <= 6.7 ? y : 13.4 - y;
        depth[d < 2.2 ? "后场" : d < 4.47 ? "中场" : "前场"]++;
        if (prev) {
          const step = Math.hypot(x - prev[0], y - prev[1]);
          const dt = (f.t_ms - prev[2]) / 1000;
          if (dt > 0 && dt <= 0.5 && step / dt <= 8) dist_m += step;
        }
        prev = [x, y, f.t_ms];
      }
    }
    const won = decided.filter((r) => r.winner === pid).length;
    const sub = (sel: (n: number) => boolean) => decided.filter((r) => sel(r.shot_ids.length));
    const short = sub((n) => n <= 4);
    const long = sub((n) => n >= 7);
    return {
      player_id: pid,
      metrics: [
        metric(`${pid}.shots`, "击球数", shots.length, "拍", known, 30, 8),
        metric(`${pid}.shot_share`, "击球占比", pct(shots.length, known), "%", known, 30, 8),
        metric(`${pid}.move.distance_m`, "回合内移动距离", samples ? dist_m : null, "m", samples, 300, 60, MOCK_NOTE),
        metric(`${pid}.move.per_rally_m`, "每回合平均移动", samples && nr ? dist_m / nr : null, "m", nr, 10, 3, MOCK_NOTE),
        metric(`${pid}.win.won`, "得分回合数", decided.length ? won : null, "分", decided.length, 15, 5),
        metric(`${pid}.win.rate_pct`, "回合胜率", pct(won, decided.length), "%", decided.length, 15, 5),
        metric(`${pid}.win.short_rate_pct`, "短回合（≤4拍）胜率", pct(short.filter((r) => r.winner === pid).length, short.length), "%", short.length, 10, 4),
        metric(`${pid}.win.long_rate_pct`, "长回合（≥7拍）胜率", pct(long.filter((r) => r.winner === pid).length, long.length), "%", long.length, 10, 4),
      ],
      distributions: [
        dist(`${pid}.pos.depth_zones`, "回合内站位：前中后场（帧数）", depth, 300, 60, MOCK_NOTE),
        { ...dist(`${pid}.shot_types`, "击球类型分布", { 未分类: shots.length }, 30, 8, "击球类型分类尚未接入，暂不可判定"), status: "undeterminable", confidence: "low" },
      ],
    };
  });

  const buckets: Record<string, number> = { "1-3拍": 0, "4-6拍": 0, "7-10拍": 0, "11拍以上": 0 };
  lens.forEach((n) => buckets[n <= 3 ? "1-3拍" : n <= 6 ? "4-6拍" : n <= 10 ? "7-10拍" : "11拍以上"]++);
  const caveats = [MOCK_NOTE + "，与后端算法一致但只覆盖部分指标。"];
  if (ev.shots.length && confirmed / ev.shots.length < 0.5) caveats.push(`只有 ${confirmed}/${ev.shots.length} 拍经过人工确认，统计基于自动识别结果。`);
  if (decided.length < nr) caveats.push(`${nr} 个回合中有 ${decided.length} 个标注了胜负；胜负相关的结论只基于这些回合。`);
  caveats.push("击球类型分类尚未接入，暂不评价具体技术动作或球路选择。");

  return {
    schema_version: "0.1.0",
    video_id: ev.video_id,
    fingerprint: hashStr(JSON.stringify(ev)),
    duration_ms: overlay.video.duration_ms,
    calibration_status: overlay.court[0]?.status ?? null,
    shot_types_available: false,
    metrics: [
      metric("match.rallies", "回合数", nr, "个", nr, 10, 3),
      metric("match.shots", "击球数", ev.shots.length, "拍", ev.shots.length, 50, 10),
      metric("match.rally_len.mean", "平均每回合拍数", mean(lens), "拍", nr, 10, 3),
      metric("match.rally_len.max", "最长回合拍数", lens.length ? Math.max(...lens) : null, "拍", nr, 10, 3),
      metric("match.rally_dur.mean_s", "平均回合时长", mean(durs), "s", nr, 10, 3),
      metric("match.confirmed_shots_pct", "已人工确认的击球比例", pct(confirmed, ev.shots.length), "%", ev.shots.length, 1, 1),
      metric("match.winners_known", "已标注胜负的回合数", decided.length, "个", nr, 1, 1),
    ],
    distributions: [dist("match.rally_len.buckets", "回合长度分布（回合数）", buckets, 10, 3)],
    players,
    caveats,
  };
}

export function mockCoach(stats: MatchStats): CoachReport {
  const idx = indexStats(stats);
  const m = (k: string) => {
    const e = idx.get(k);
    return e && isMetric(e) && e.value !== null && e.status !== "undeterminable" ? e : null;
  };
  const insights: CoachInsight[] = [];
  const mean = m("match.rally_len.mean");
  if (mean) {
    insights.push({
      kind: "pattern", player_id: null, title: mean.value! <= 5 ? "本场以短回合为主" : "本场多拍相持较多",
      detail: `平均每回合 ${fmtValue(mean)}（${mean.n} 个回合）。`, evidence: [mean.key], confidence: mean.confidence, grounded: true, warnings: [],
    });
  }
  for (const pid of ["P1", "P2"] as const) {
    const rate = m(`${pid}.win.rate_pct`);
    if (rate) {
      insights.push({
        kind: rate.value! >= 50 ? "strength" : "weakness", player_id: pid, title: `${pid} 回合胜率 ${fmtValue(rate)}`,
        detail: `基于 ${rate.n} 个已标注胜负的回合${rate.confidence === "high" ? "" : "，样本较少，仅供参考"}。`,
        evidence: [rate.key], confidence: rate.confidence, grounded: true, warnings: [],
      });
    }
  }
  const order: MetricConfidence[] = ["low", "medium", "high"];
  const a = m("P1.move.per_rally_m");
  const b = m("P2.move.per_rally_m");
  if (a && b && Math.max(a.value!, b.value!) / Math.max(1e-6, Math.min(a.value!, b.value!)) >= 1.2) {
    const [hi, lo] = a.value! >= b.value! ? [a, b] : [b, a];
    const pid = hi.key.split(".")[0] as PlayerId;
    insights.push({
      kind: "pattern", player_id: pid, title: `${pid} 跑动明显更多`,
      detail: `每回合平均移动 ${fmtValue(hi)}，对手 ${fmtValue(lo)}；跑动多可能说明被调动。`, evidence: [hi.key, lo.key],
      confidence: order[Math.min(order.indexOf(hi.confidence), order.indexOf(lo.confidence))], grounded: true, warnings: [],
    });
  }
  insights.push({
    kind: "data_quality", player_id: null, title: "暂不评价技术动作", detail: "击球类型分类尚未接入，本报告不涉及拍法和球路选择。",
    evidence: [], confidence: "high", grounded: true, warnings: [],
  });
  return {
    schema_version: "0.1.0", video_id: stats.video_id, stats_fingerprint: stats.fingerprint, provider: "offline", model: "mock",
    created_at: new Date().toISOString(), language: "zh",
    summary: `演示模式下用规则模板生成的点评，共 ${insights.length - 1} 条数据观察。连接后端并配置 Claude API 后可得到完整的 AI 教练分析。`,
    insights, training: [], caveats: stats.caveats, fallback_reason: "演示模式不调用 AI 模型", dropped: 0,
  };
}
