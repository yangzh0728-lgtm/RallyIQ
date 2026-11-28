// Synthetic long-term profile for demo mode (?mock), so the profile page can be shown without a backend.
//
// The numbers are invented, but nothing is typed in twice: each match below lists raw per-match samples
// (wins / rallies, metres, frames), and the profile is derived from them with the same rules as
// rallyiq/memory/profile.py and rallyiq/memory/opponent.py (status from sample size, latest / previous /
// delta / trend from the last two usable points, retests against the next match). So the table, the
// change sentences and the retest rows cannot disagree with each other.
//
// Everything here is a pure function of ``now`` (ms since epoch).
import { generateMock } from "./mock";
import type {
  Distribution,
  Habit,
  HabitEvidence,
  MetricConfidence,
  NoteRef,
  OpponentReport,
  PersonSummary,
  PlayerId,
  PlayerProfile,
  ProfileMetric,
  ProfileVideo,
  RetestResult,
  ShotEvent,
  ShotType,
  Status,
} from "./types";

export const DEMO_SELF = "我";
export const DEMO_OPPONENT = "小林";
/** First line of every demo profile and opponent report. */
const DEMO_NOTICE = "演示数据：以下档案为合成示例，连接后端后会换成你确认过的真实比赛";

// Thresholds of rallyiq/memory/profile.py.
const FLAT_REL = 0.1;
const CONFLICT_REL = 0.35;
const HEADLINE = ["win.rate_pct", "move.per_rally_m", "cover.ratio_pct", "pos.depth_mean_m", "shots", "shot_share"];
const CONFLICT_NOTE = "与上一场差异较大，可能是对手或场地条件不同，需要更多比赛确认";
/** Appended to the log by the demo history: 6 commits (one per named player) + one withdrawal and one re-commit. */
const LOG_EVENTS = 8;

const DAY = 86_400_000;
const HOUR = 3_600_000;
const round = (v: number, digits: number) => Math.round(v * 10 ** digits) / 10 ** digits;
const rel = (a: number, b: number) => Math.abs(a - b) / Math.max(Math.abs(b), 1e-9);
/** Python's ``{:g}``: up to six significant digits, no trailing zeros. */
const g = (v: number) => String(Number(v.toPrecision(6)));

// ------------------------------------------------------------------ per-match samples

/** label, unit and the sample sizes for "ok" / "needs_review" (rallyiq/report/stats.py and doubles.py). */
const DEFS: Record<string, { label: string; unit: string; ok: number; review: number }> = {
  "win.rate_pct": { label: "回合胜率", unit: "%", ok: 15, review: 5 },
  "move.per_rally_m": { label: "每回合平均移动", unit: "m", ok: 10, review: 3 },
  "cover.ratio_pct": { label: "覆盖率（占单打半场）", unit: "%", ok: 300, review: 60 },
  shot_share: { label: "击球占比", unit: "%", ok: 30, review: 8 },
  "dbl.front_pct": { label: "前后站位时在前场的比例", unit: "%", ok: 150, review: 30 },
  "dbl.smash_share_pct": { label: "占本队杀球的比例", unit: "%", ok: 15, review: 5 },
  "win.receive_rate_pct": { label: "接发球回合胜率", unit: "%", ok: 10, review: 4 },
  "win.serve_rate_pct": { label: "发球回合胜率", unit: "%", ok: 10, review: 4 },
};

interface Sample {
  key: string;
  value: number | null;
  n: number;
  status: Status;
  confidence: MetricConfidence;
}

/** One measured value. ``degrade`` drops one level, as when the court calibration is not confirmed. */
function measured(key: string, value: number | null, n: number, degrade = false): Sample {
  const d = DEFS[key];
  const size = value == null ? 0 : n;
  if (size <= 0) return { key, value: null, n, status: "undeterminable", confidence: "low" };
  let status: Status = size >= d.ok ? "ok" : "needs_review";
  let confidence: MetricConfidence = size >= d.ok ? "high" : size >= d.review ? "medium" : "low";
  if (degrade) {
    status = "needs_review";
    confidence = confidence === "high" ? "medium" : "low";
  }
  return { key, value, n, status, confidence };
}
/** A percentage of ``hits`` out of ``n``, so the value always matches its sample. */
const share = (key: string, hits: number, n: number) => measured(key, n ? round((100 * hits) / n, 1) : null, n);
/** A metric that could not be measured in that match (shown as "—"). */
const missing = (key: string) => measured(key, null, 0);

interface Training {
  drill: string;
  why: string;
  focus: string | null;
  retest_metric: string;
  sources: NoteRef[];
}

/** What one committed match contributes to one person's profile (the demo's stand-in for AnalysisEvent). */
interface MatchEvent {
  video: ProfileVideo;
  metrics: Sample[];
  training: Training[];
}

const note = (id: string, title: string, t_s: number, drill: string): NoteRef => ({
  note_id: `demo-${id}`,
  coach: "示例教练 (合成数据)",
  title,
  url: `https://example.com/rallyiq-demo/${id}`,
  t_s,
  drill,
});

const OBLIQUE = { camera_view: "end_oblique", camera_height: "low", fps: "50+", resolution: "1080p" };
// Same as mockQuality() and the 1280×720, 30 fps demo clips.
const DEMO_CAMERA = { camera_view: "end_center", camera_height: "medium", fps: "25-35", resolution: "720p" };

/** The four demo matches, oldest first. Only the last two exist in the demo library. */
function matches(now: number) {
  const at = (daysAgo: number, hour: number, minute: number) => {
    const d = new Date(now - daysAgo * DAY);
    d.setHours(hour, minute, 0, 0);
    return d.toISOString();
  };
  const video = (video_id: string, filename: string, recorded_at: string, conditions: Record<string, string>) => ({ video_id, filename, recorded_at, conditions });
  return {
    club: video("demo_club_singles", "周六俱乐部单打（合成数据）.mp4", at(41, 19, 40), { ...OBLIQUE, light: "normal", match_format: "singles" }),
    practice: video("demo_practice_singles", "周中练习赛单打（合成数据）.mp4", at(27, 20, 5), { ...OBLIQUE, light: "dark", match_format: "singles" }),
    doubles: video("demo_doubles", "示例双打（合成数据）.mp4", at(13, 20, 15), { ...DEMO_CAMERA, match_format: "doubles" }),
    // The demo library lists this clip as uploaded one hour ago.
    singles: video("demo", "示例单打（合成数据）.mp4", new Date(now - HOUR).toISOString(), { ...DEMO_CAMERA, match_format: "singles" }),
  };
}

const as = (v: Omit<ProfileVideo, "player_id" | "opponents">, player_id: PlayerId, opponents: string[]): ProfileVideo => ({ ...v, player_id, opponents });

/** The user's four matches. Serve + receive rallies add up to the decided rallies of each match. */
function selfEvents(now: number): MatchEvent[] {
  const m = matches(now);
  return [
    {
      video: as(m.club, "P1", [DEMO_OPPONENT]),
      metrics: [
        share("win.rate_pct", 16, 37),
        measured("move.per_rally_m", 13.8, 41, true),
        measured("cover.ratio_pct", 58.4, 6120, true),
        share("shot_share", 204, 412),
        share("win.receive_rate_pct", 8, 18),
        share("win.serve_rate_pct", 8, 19),
      ],
      training: [
        {
          drill: "发球后第三拍抢网",
          why: "发球回合胜率只有 42.1%：发网前球后常停在原地，第三拍被动起球",
          focus: "发球后的第三拍",
          retest_metric: "win.serve_rate_pct",
          sources: [note("third-shot", "发球后的站位与第三拍启动（示例）", 95, "发球后第三拍抢网")],
        },
      ],
    },
    {
      video: as(m.practice, "P1", []),
      metrics: [
        share("win.rate_pct", 19, 41),
        measured("move.per_rally_m", 13.1, 44, true),
        measured("cover.ratio_pct", 61, 6890, true),
        share("shot_share", 221, 437),
        share("win.receive_rate_pct", 9, 21),
        share("win.serve_rate_pct", 10, 20),
      ],
      training: [
        {
          drill: "接发球推压后场两角",
          why: "接发球回合胜率 42.9%：接发后多为挑高球，把主动让给对手",
          focus: "接发球起球过多",
          retest_metric: "win.receive_rate_pct",
          sources: [],
        },
        {
          drill: "米字步法：每拍回中",
          why: "每回合平均移动 13.1 m：击球后回中慢，被连续调动",
          focus: "击球后回中",
          retest_metric: "move.per_rally_m",
          sources: [],
        },
      ],
    },
    {
      // Doubles: no usable player tracks in this one, so movement and formation could not be measured.
      video: as(m.doubles, "P1", []),
      metrics: [
        share("win.rate_pct", 20, 42),
        missing("move.per_rally_m"),
        missing("cover.ratio_pct"),
        share("shot_share", 99, 376),
        missing("dbl.front_pct"),
        share("dbl.smash_share_pct", 15, 24),
        share("win.receive_rate_pct", 8, 22),
        share("win.serve_rate_pct", 12, 20),
      ],
      training: [],
    },
    {
      video: as(m.singles, "P1", [DEMO_OPPONENT]),
      metrics: [
        share("win.rate_pct", 22, 40),
        measured("move.per_rally_m", 12.9, 43),
        measured("cover.ratio_pct", 52.7, 6435),
        share("shot_share", 196, 398),
        share("win.receive_rate_pct", 10, 21),
        share("win.serve_rate_pct", 12, 19),
      ],
      training: [
        {
          drill: "多球四角步法（重点两个后场角）",
          why: "覆盖率 52.7%，比上一场单打低 8.3 个百分点：后场两角到位偏晚",
          focus: "后场两角到位",
          retest_metric: "cover.ratio_pct",
          sources: [note("four-corners", "四角步法的启动与回中（示例）", 210, "多球四角步法")],
        },
      ],
    },
  ];
}

/** The opponent's side of the two singles matches against the user (the complement of the numbers above). */
function opponentEvents(now: number): MatchEvent[] {
  const m = matches(now);
  return [
    {
      video: as(m.club, "P2", [DEMO_SELF]),
      metrics: [
        share("win.rate_pct", 21, 37),
        measured("move.per_rally_m", 12.4, 41, true),
        measured("cover.ratio_pct", 63.1, 6120, true),
        share("shot_share", 208, 412),
        share("win.receive_rate_pct", 11, 19),
        share("win.serve_rate_pct", 10, 18),
      ],
      training: [],
    },
    {
      video: as(m.singles, "P2", [DEMO_SELF]),
      metrics: [
        share("win.rate_pct", 18, 40),
        measured("move.per_rally_m", 14.2, 43),
        measured("cover.ratio_pct", 64.9, 6435),
        share("shot_share", 202, 398),
        share("win.receive_rate_pct", 7, 19),
        share("win.serve_rate_pct", 11, 21),
      ],
      training: [],
    },
  ];
}

// ------------------------------------------------------------------ profile (rallyiq/memory/profile.py)

function buildMetrics(events: MatchEvent[]): ProfileMetric[] {
  const series = new Map<string, { e: MatchEvent; s: Sample }[]>();
  for (const e of events) {
    for (const s of e.metrics) {
      if (!series.has(s.key)) series.set(s.key, []);
      series.get(s.key)!.push({ e, s });
    }
  }
  const metrics: ProfileMetric[] = [];
  for (const [key, pts] of series) {
    const usable = pts.filter((p) => p.s.value != null && p.s.status !== "undeterminable");
    const last2 = usable.slice(-2);
    const latest = usable.length ? usable[usable.length - 1].s.value : null;
    const previous = usable.length >= 2 ? usable[usable.length - 2].s.value : null;
    let trend: ProfileMetric["trend"];
    if (latest == null) trend = "insufficient";
    else if (previous == null) trend = "new";
    else if (rel(latest, previous) < FLAT_REL) trend = "flat";
    else trend = latest > previous ? "up" : "down";
    const conflict = latest != null && previous != null && rel(latest, previous) > CONFLICT_REL && last2.every((p) => p.s.status === "ok");
    const order: MetricConfidence[] = ["low", "medium", "high"];
    const confidence = last2.length ? order[Math.min(...last2.map((p) => order.indexOf(p.s.confidence)))] : "low";
    metrics.push({
      key,
      label: DEFS[key].label,
      unit: DEFS[key].unit,
      points: pts.map(({ e, s }) => ({ video_id: e.video.video_id, filename: e.video.filename, recorded_at: e.video.recorded_at, value: s.value, n: s.n, status: s.status })),
      latest,
      previous,
      delta: latest != null && previous != null ? round(latest - previous, 3) : null,
      trend,
      confidence,
      note: conflict ? CONFLICT_NOTE : null,
    });
  }
  const rank = (key: string) => (HEADLINE.includes(key) ? HEADLINE.indexOf(key) : HEADLINE.length);
  return metrics.sort((a, b) => rank(a.key) - rank(b.key) || (a.key < b.key ? -1 : a.key > b.key ? 1 : 0));
}

function buildChanges(events: MatchEvent[], metrics: ProfileMetric[]): string[] {
  const changes: string[] = [];
  for (const m of metrics) {
    if (!HEADLINE.includes(m.key) || m.latest == null) continue;
    const unit = m.unit ?? "";
    if (m.trend === "new") {
      changes.push(`新增：${m.label} ${g(m.latest)}${unit}（第一场有效数据，n=${m.points[m.points.length - 1].n}）`);
    } else if ((m.trend === "up" || m.trend === "down") && m.previous != null) {
      const line = `趋势：${m.label} ${m.trend === "up" ? "上升" : "下降"}，${g(m.previous)} → ${g(m.latest)}${unit}`;
      changes.push(line + (m.note ? `（${m.note}）` : ""));
    }
  }
  if (events.length >= 2) {
    const a = events[events.length - 2].video.conditions;
    const b = events[events.length - 1].video.conditions;
    const diff = ["camera_view", "fps"].filter((k) => a[k] && b[k] && a[k] !== b[k]);
    if (diff.length) changes.push(`冲突提醒：最近两场的拍摄条件不同（${diff.join("、")}），对比结论只能作参考`);
  }
  return changes;
}

/** Each training item of one match, measured on the next committed match. */
function buildRetests(events: MatchEvent[]): RetestResult[] {
  const out: RetestResult[] = [];
  events.forEach((e, i) => {
    const next = i + 1 < events.length ? events[i + 1] : null;
    for (const t of e.training) {
      const before = e.metrics.find((s) => s.key === t.retest_metric) ?? null;
      const after = next?.metrics.find((s) => s.key === t.retest_metric) ?? null;
      const same = next ? ["camera_view", "fps"].every((k) => e.video.conditions[k] === next.video.conditions[k]) : null;
      let status: RetestResult["status"];
      if (!next) status = "pending";
      else if (before?.value == null || after?.value == null) status = "not_measurable";
      else if (rel(after.value, before.value) < FLAT_REL) status = "unchanged";
      // "Better" is only clear for win and coverage rates; elsewhere the change is reported as such.
      else if (t.retest_metric.startsWith("win.") || t.retest_metric.startsWith("cover.")) status = after.value > before.value ? "improved" : "worse";
      else status = "changed";
      out.push({
        drill: t.drill,
        why: t.why,
        focus: t.focus,
        sources: t.sources,
        metric: t.retest_metric,
        label: DEFS[t.retest_metric]?.label ?? null,
        unit: DEFS[t.retest_metric]?.unit ?? null,
        from_video_id: e.video.video_id,
        to_video_id: next ? next.video.video_id : null,
        before: before?.value ?? null,
        after: after?.value ?? null,
        delta: before?.value != null && after?.value != null ? round(after.value - before.value, 3) : null,
        same_conditions: same,
        status,
      });
    }
  });
  return out;
}

function buildProfile(person: string, isSelf: boolean, events: MatchEvent[], notice: string): PlayerProfile {
  const metrics = buildMetrics(events);
  return {
    person,
    is_self: isSelf,
    videos: events.map((e) => e.video),
    metrics,
    retests: buildRetests(events),
    changes: [notice, ...buildChanges(events, metrics)],
    log_events: LOG_EVENTS,
  };
}

/** The user's own profile: four matches over six weeks, the latest being the demo singles clip. */
export function demoSelfProfile(now: number): PlayerProfile {
  return buildProfile(
    DEMO_SELF,
    true,
    selfEvents(now),
    `${DEMO_NOTICE}。数值按整场比赛的量级编写，和 75 秒示例视频的赛后报告对不上；较早两场没有可打开的视频。`,
  );
}

/** The opponent's profile, built from the same two singles matches seen from the other side. */
export function demoOpponentProfile(now: number): PlayerProfile {
  return buildProfile(
    DEMO_OPPONENT,
    false,
    opponentEvents(now),
    `${DEMO_NOTICE}。数值按整场比赛的量级编写，和 75 秒示例视频的赛后报告对不上；较早一场没有可打开的视频。`,
  );
}

/** Everyone named in the demo log, the user first. */
export function demoPeople(now: number): PersonSummary[] {
  const entry = (person: string, is_self: boolean, events: MatchEvent[]): PersonSummary => ({
    person,
    videos: events.length,
    is_self,
    last: events.length ? events[events.length - 1].video.recorded_at : null,
  });
  return [entry(DEMO_SELF, true, selfEvents(now)), entry(DEMO_OPPONENT, false, opponentEvents(now))];
}

// ------------------------------------------------------------------ opponent report (rallyiq/memory/opponent.py)

/** Confidence of a habit from its sample size and share. */
const habitConfidence = (n: number, p: number): Habit["confidence"] => (n >= 30 && p >= 0.6 ? "high" : n >= 12 ? "medium" : "low");

function habit(condition: string, outcome: string, hits: number, n: number, evidence: HabitEvidence[], counter_examples: HabitEvidence[]): Habit {
  const p = round(hits / n, 3);
  return { condition, outcome, p, n, hits, confidence: habitConfidence(n, p), evidence, counter_examples };
}

/**
 * Evidence is picked from the opponent's (P2) own shots in the demo singles clip by stroke type, so every
 * timecode opens a real moment of the 75 s video and the stroke shown there is the one the habit names.
 */
function demoEvidence() {
  const all = generateMock("demo").events.shots;
  const typed = all.filter((s) => s.hitter === "P2" && s.shot_type);
  const pick = (keep: (s: ShotEvent, prev: ShotEvent | undefined) => boolean): HabitEvidence[] =>
    typed
      .filter((s) => keep(s, all[all.indexOf(s) - 1]))
      .slice(0, 5)
      .map((s) => ({ video_id: "demo", shot_id: s.shot_id, t_ms: s.t_ms }));
  const is = (...types: ShotType[]) => (s: ShotEvent) => !!s.shot_type && types.includes(s.shot_type);
  /** The opponent's replies to one of our strokes in the same rally, split by whether they were ``reply``. */
  const after = (ours: ShotType, reply: ShotType, same: boolean) =>
    pick((s, prev) => !!prev && prev.rally_id === s.rally_id && prev.hitter === "P1" && prev.shot_type === ours && (s.shot_type === reply) === same);
  return { pick, is, after };
}

export function demoOpponentReport(now: number): OpponentReport {
  const events = opponentEvents(now);
  const { pick, is, after } = demoEvidence();
  const typeCounts: Record<string, number> = {
    杀球: 82, 挑球: 49, 放网: 44, 高远球: 41, 吊球: 35, 发网前球: 31, 平抽: 22, 推球: 19, 勾对角: 12, 发后场球: 9, 扑球: 8,
  };
  const zoneCounts: Record<string, number> = { 后场: 172, 前场: 130, 中场: 96 };
  const sum = (c: Record<string, number>) => Object.values(c).reduce((a, b) => a + b, 0);
  const shots = 208 + 202; // the opponent's share of the shots in the two matches above
  const typed = sum(typeCounts);
  const reviewed = 214;
  const distributions: Distribution[] = [
    { key: "shot_type", label: "击球类型分布", counts: typeCounts, n: typed, status: "needs_review", confidence: habitConfidence(typed, 1), note: null },
    { key: "zone", label: "击球位置（本方半场）", counts: zoneCounts, n: sum(zoneCounts), status: "needs_review", confidence: habitConfidence(sum(zoneCounts), 1), note: null },
  ];
  return {
    person: DEMO_OPPONENT,
    videos: events.map((e) => e.video),
    shots,
    typed_shots: typed,
    distributions,
    habits: [
      habit("发球时", "发网前球", typeCounts["发网前球"], typeCounts["发网前球"] + typeCounts["发后场球"], pick(is("short_serve")), pick(is("long_serve"))),
      habit("在后场击球时", "杀球", 76, 152, pick(is("smash")), pick(is("clear", "drop"))),
      habit("在前场击球时", "挑球", 49, 116, pick(is("lift")), pick(is("net_shot", "cross_net", "net_kill"))),
    ],
    next_shot: [
      habit("我方发网前球之后", "挑球", 21, 33, after("short_serve", "lift", true), after("short_serve", "lift", false)),
      habit("我方推球之后", "挑球", 5, 11, after("push", "lift", true), after("push", "lift", false)),
    ],
    caveats: [
      `${DEMO_NOTICE}。视频证据都指向示例单打里这位对手（P2）的击球。`,
      `只有 ${events.length} 场比赛的数据，习惯判断样本有限`,
      `${typed - reviewed}/${typed} 拍的击球类型仍是待复核的模型判断（规则 v1）；在工作台确认或修正类型后重新写入档案会更可靠`,
    ],
  };
}
