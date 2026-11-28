import { Fragment, useEffect, useMemo, useState } from "react";
import { compareRows, leadSeries, type CompareRow } from "../lib/flow";
import { colorOf, labelOf } from "../lib/players";
import { fmtValue, STATUS_LABEL } from "../lib/report";
import { formatMs } from "../lib/corrections";
import type { GameScore, MatchStats, Metric, RallyLine } from "../lib/types";
import { Empty, Icon, SideBadge } from "./ui";

const sideColor = (s: string | null | undefined) => (s ? colorOf(s) : "var(--surface-4)");
const at = (videoId: string, ms: number) => `#/videos/${encodeURIComponent(videoId)}?t=${ms}`;

// ---------------------------------------------------------------- metric groups (shared with the report page)

export type GroupId = "rally" | "score" | "review" | "hit" | "win" | "move" | "tempo" | "dbl" | "style" | "footwork" | "other";
export const GROUP_ORDER: GroupId[] = ["rally", "score", "review", "hit", "win", "move", "tempo", "dbl", "style", "footwork", "other"];
export const GROUP_LABEL: Record<GroupId, string> = {
  rally: "回合",
  score: "比分",
  review: "标注与复核",
  hit: "击球",
  win: "得分",
  move: "移动与站位",
  tempo: "节奏",
  dbl: "双打配合",
  style: "打法特征",
  footwork: "步法",
  other: "其他",
};

/** Which group a metric / distribution key belongs to: "P1.win.rate_pct" -> win, "match.tempo.x" -> tempo. */
export function groupOf(key: string): GroupId {
  const [subject, a = ""] = key.split(".");
  if (subject === "match") {
    if (a === "tempo") return "tempo";
    if (a === "score") return "score";
    if (a.startsWith("confirmed") || a === "winners_known") return "review";
    if (a === "rallies" || a === "shots" || a.startsWith("rally")) return "rally";
    return "other";
  }
  switch (a) {
    case "shots":
    case "shot_share":
    case "shot_types":
    case "hit":
      return "hit";
    case "win":
    case "lost":
      return "win";
    case "move":
    case "cover":
    case "pos":
      return "move";
    case "tempo":
    case "style":
    case "dbl":
    case "footwork":
      return a;
    default:
      return "other";
  }
}

/** The handful of per-player numbers shown first (key without the player prefix), in display order. */
export const HEAD_PLAYER = ["shots", "shot_share", "move.distance_m", "move.per_rally_m", "win.won", "win.rate_pct", "win.short_rate_pct", "win.long_rate_pct"];

/** Long lists start short; this button shows the rest (and folds it again). */
export function MoreButton({ open, onToggle, more, less = "收起" }: { open: boolean; onToggle: () => void; more: string; less?: string }) {
  return (
    <div className="more-row">
      <button className="ghost small more-btn" aria-expanded={open} onClick={onToggle}>
        {open ? less : more}
        <Icon name="chevronDown" className={open ? "flip" : ""} />
      </button>
    </div>
  );
}

function Who({ side }: { side: string | null }) {
  if (!side) return <span className="muted">—</span>;
  return (
    <span className="who">
      <span className="dot" style={{ background: sideColor(side) }} />
      {side}
    </span>
  );
}

/** Width of an element in CSS px, so the chart can be drawn 1:1 (text and bars keep their size). */
function useWidth(fallback: number) {
  const [el, setEl] = useState<HTMLDivElement | null>(null);
  const [width, setWidth] = useState(fallback);
  useEffect(() => {
    if (!el) return;
    const measure = () => setWidth(Math.max(280, Math.round(el.clientWidth)));
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, [el]);
  return [setEl, width] as const;
}

/** A bar from the baseline to ``y1``: rounded at the data end, square at the baseline. */
function barPath(x: number, w: number, base: number, tip: number): string {
  const r = Math.min(2, w / 2, Math.abs(tip - base));
  const s = tip < base ? 1 : -1; // +1: grows upward
  return `M${x},${base}V${tip + s * r}Q${x},${tip} ${x + r},${tip}H${x + w - r}Q${x + w},${tip} ${x + w},${tip + s * r}V${base}Z`;
}

/** Games, lead after every decided rally, and the flow metrics (runs, lead changes). */
export function ScoreFlow({ stats, id }: { stats: MatchStats; id?: string }) {
  const [chartRef, W] = useWidth(720);
  const rallies = stats.rallies ?? [];
  const games = stats.games ?? [];
  const decided = rallies.find((r) => r.score);
  if (!decided) {
    return (
      <section id={id} className="card">
        <div className="card-head">
          <h2>比分走势</h2>
        </div>
        <Empty icon="trend" title="还没有标注回合胜负">
          在工作台的修正面板里给回合设置胜者后，这里会按 21 分制推算每局比分、领先变化和连续得分。
        </Empty>
      </section>
    );
  }
  const sides = Object.keys(decided.score!) as [string, string];
  const series = leadSeries(rallies, sides);
  const top = Math.max(3, ...series.map((s) => Math.abs(s.lead)));
  const padL = 30;
  const padR = 4;
  const padT = 18;
  const padB = 20;
  const H = 176;
  const mid = padT + (H - padT - padB) / 2;
  const half = (H - padT - padB) / 2;
  // Bars at most 22 px wide, so a short match does not turn into two giant blocks.
  const bw = Math.min((W - padL - padR) / Math.max(1, series.length), 22);
  const gap = Math.min(2, bw * 0.25);
  const labelEvery = Math.max(1, Math.ceil(20 / bw));
  const find = (key: string): Metric | undefined =>
    [...stats.metrics, ...stats.players.flatMap((p) => p.metrics)].find((m) => m.key === key);
  const facts: string[] = [];
  const lc = find("match.score.lead_changes");
  if (lc?.value != null) facts.push(`领先易手 ${lc.value} 次`);
  for (const p of stats.players) {
    const run = find(`${p.player_id}.win.longest_run`);
    if (run?.value) facts.push(`${p.player_id} 最长连续得分 ${run.value} 分`);
  }
  const undecided = rallies.length - rallies.filter((r) => r.score).length;
  const multiGame = series.some((s, i) => i > 0 && series[i - 1].rally.game_no !== s.rally.game_no);
  return (
    <section id={id} className="card">
      <div className="card-head">
        <h2>比分走势</h2>
        <span className="sub">
          领先分差：向上 {sides[0]} 领先，向下 {sides[1]} 领先{multiGame ? "；竖虚线是换局" : ""}
        </span>
        <span className="spacer" />
        <div className="legend">
          {sides.map((s) => (
            <Who key={s} side={s} />
          ))}
          {multiGame && (
            <span>
              <span className="key-dash" />
              换局
            </span>
          )}
        </div>
      </div>
      {games.length > 0 && (
        <div className="flow-games">
          {games.map((g: GameScore) => (
            <div key={g.game_no} className="flow-game">
              <div className="mono value">
                {g.score[sides[0]] ?? 0}
                <i>:</i>
                {g.score[sides[1]] ?? 0}
              </div>
              <div className="small muted game-line">
                第 {g.game_no} 局 · {g.winner ? <Who side={g.winner} /> : "未完成"}
                {g.winner ? " 胜" : ""}
              </div>
            </div>
          ))}
        </div>
      )}
      <div className="flow-wrap" ref={chartRef}>
        <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} className="flow-chart" role="img" aria-label="比分走势">
          <line x1={padL} x2={W - padR} y1={mid - half} y2={mid - half} className="grid" />
          <line x1={padL} x2={W - padR} y1={mid + half} y2={mid + half} className="grid" />
          <text x={padL - 8} y={mid - half + 3.5} className="axis" textAnchor="end">+{top}</text>
          <text x={padL - 8} y={mid + 3.5} className="axis" textAnchor="end">0</text>
          <text x={padL - 8} y={mid + half + 3.5} className="axis" textAnchor="end">−{top}</text>
          <text x={padL - 8} y={H - 4} className="axis" textAnchor="end">回合</text>
          {series.map(({ rally, lead }, i) => {
            const x = padL + i * bw;
            const h = Math.max((half * Math.abs(lead)) / top, 1.5);
            const w = Math.max(bw - gap, 1);
            const newGame = i > 0 && series[i - 1].rally.game_no !== rally.game_no;
            const sc = `${rally.score![sides[0]]}-${rally.score![sides[1]]}`;
            return (
              <g key={rally.rally_id}>
                {newGame && (
                  <>
                    <line x1={x} x2={x} y1={padT - 6} y2={H - padB + 4} className="game-line" />
                    {x + 44 > W - padR ? (
                      <text x={x - 4} y={10} className="axis" textAnchor="end">第 {rally.game_no} 局</text>
                    ) : (
                      <text x={x + 4} y={10} className="axis">第 {rally.game_no} 局</text>
                    )}
                  </>
                )}
                <path
                  className="bar-mark"
                  d={barPath(x + gap / 2, w, mid, lead >= 0 ? mid - h : mid + h)}
                  fill={lead > 0 ? sideColor(sides[0]) : lead < 0 ? sideColor(sides[1]) : "var(--muted)"}
                >
                  <title>{`第 ${rally.rally_no} 回合 · ${formatMs(rally.start_ms)} · ${rally.winner} 得分 · 比分 ${sc}`}</title>
                </path>
                {(i % labelEvery === 0 || i === series.length - 1) && (i === series.length - 1 || series.length - 1 - i >= labelEvery) && (
                  <text x={x + bw / 2} y={H - 4} className="axis" textAnchor="middle">{rally.rally_no}</text>
                )}
              </g>
            );
          })}
          <line x1={padL} x2={W - padR} y1={mid} y2={mid} className="baseline" />
        </svg>
      </div>
      <div className="card-foot">
        按已标注的回合胜负推算（21 分制，领先 2 分获胜，30 分封顶）。{facts.join("；")}
        {facts.length ? "。" : ""}
        {undecided > 0 && `还有 ${undecided} 个回合没有标注胜负，比分可能偏低。`}
      </div>
    </section>
  );
}

type Filter = "all" | "long" | "short" | string;
const RALLY_ROWS = 12;
const COMPARE_ROWS = 12;

/** One row per rally: time (jumps to the video), shots, duration, server, winner, score, last shot, tempo. */
export function RallyTable({ stats, videoId, id }: { stats: MatchStats; videoId: string; id?: string }) {
  const rallies = stats.rallies ?? [];
  const [filter, setFilter] = useState<Filter>("all");
  const [expanded, setExpanded] = useState(false);
  const sides = useMemo(() => [...new Set(rallies.map((r) => r.winner).filter((w): w is NonNullable<typeof w> => !!w))].sort(), [rallies]);
  if (!rallies.length) return null;
  const maxShots = Math.max(1, ...rallies.map((r) => r.shots));
  const shown = rallies.filter((r: RallyLine) =>
    filter === "all" ? true : filter === "long" ? r.shots >= 7 : filter === "short" ? r.shots <= 4 : r.winner === filter,
  );
  // A full match has 60+ rallies: start with the first rows, the button below shows the rest.
  const foldable = shown.length > RALLY_ROWS + 4;
  const visible = foldable && !expanded ? shown.slice(0, RALLY_ROWS) : shown;
  const half = Math.floor(rallies.length / 2);
  const mean = (xs: number[]) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null);
  const firstLen = mean(rallies.slice(0, half).map((r) => r.shots));
  const secondLen = mean(rallies.slice(half).map((r) => r.shots));
  const options: { value: Filter; label: string; title: string; side?: string }[] = [
    { value: "all", label: `全部 ${rallies.length}`, title: `全部 ${rallies.length} 个回合` },
    { value: "long", label: "长回合 ≥7 拍", title: "长回合（≥7 拍）" },
    { value: "short", label: "短回合 ≤4 拍", title: "短回合（≤4 拍）" },
    ...sides.map((s) => ({ value: s, label: `${s} 得分`, title: `${s} 得分的回合`, side: s })),
  ];
  return (
    <section id={id} className="card">
      <div className="card-head">
        <h2>逐回合</h2>
        <span className="sub">点时间跳回视频；柱长是拍数，颜色是得分方</span>
        <span className="spacer" />
        <div className="seg" role="group" aria-label="筛选回合">
          {options.map((o) => (
            <button key={o.value} className={filter === o.value ? "on" : ""} aria-pressed={filter === o.value} title={o.title} onClick={() => setFilter(o.value)}>
              {o.side && <span className="dot" style={{ background: sideColor(o.side) }} />}
              {o.label}
            </button>
          ))}
        </div>
      </div>
      {((firstLen != null && secondLen != null && half >= 3) || filter !== "all") && (
        <div className="rally-note">
          {firstLen != null && secondLen != null && half >= 3 && (
            <span>
              前半段平均每回合 {firstLen.toFixed(1)} 拍，后半段 {secondLen.toFixed(1)} 拍
              {Math.abs(secondLen - firstLen) >= 1 ? (secondLen > firstLen ? "：后半段相持更多" : "：后半段回合变短") : "：前后节奏接近"}。
            </span>
          )}
          <span className="spacer" />
          {filter !== "all" && (
            <span className="mono">
              符合条件 {shown.length} / {rallies.length} 个回合
            </span>
          )}
        </div>
      )}
      <div className="table-scroll">
        <table className="rally-table">
          <thead>
            <tr>
              <th className="num">#</th>
              <th>时间</th>
              <th>拍数</th>
              <th className="num">时长</th>
              <th>发球</th>
              <th>得分</th>
              <th className="num">比分</th>
              <th>最后一拍</th>
              <th className="num">拍间间隔</th>
            </tr>
          </thead>
          <tbody>
            {visible.map((r) => (
              <tr key={r.rally_id}>
                <td className="num mono muted">{r.rally_no}</td>
                <td>
                  <a className="tc" href={at(videoId, r.start_ms)} title="在工作台打开这个回合">
                    {formatMs(r.start_ms)}
                  </a>
                </td>
                <td>
                  <div className="rally-len">
                    <span className="mono n">{r.shots}</span>
                    <span className="len-track">
                      <i style={{ width: `${(100 * r.shots) / maxShots}%`, background: sideColor(r.winner) }} />
                    </span>
                  </div>
                </td>
                <td className="num mono">{r.duration_s.toFixed(1)} s</td>
                <td>
                  <Who side={r.server} />
                </td>
                <td>
                  <Who side={r.winner} />
                </td>
                <td className="num mono">{r.score ? Object.values(r.score).join(":") : "—"}</td>
                <td className="last-shot">
                  {r.last_hitter || r.last_shot_type ? (
                    <span className="who">
                      {r.last_hitter ? <Who side={r.last_hitter} /> : <span className="muted">?</span>}
                      {r.last_shot_type && <span className="type-chip">{r.last_shot_type}</span>}
                    </span>
                  ) : (
                    <span className="muted">—</span>
                  )}
                </td>
                <td className="num mono">{r.mean_interval_s != null ? `${r.mean_interval_s.toFixed(2)} s` : "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {!shown.length && <p className="muted small table-empty">没有符合条件的回合。</p>}
      {foldable && <MoreButton open={expanded} onToggle={() => setExpanded(!expanded)} more={`显示全部 ${shown.length} 个回合`} less={`只看前 ${RALLY_ROWS} 个`} />}
    </section>
  );
}

function Cell({ v, max, color, side }: { v: Metric | null; max: number; color: string; side: "left" | "right" }) {
  const has = !!v && v.value != null;
  const dim = has && v.status !== "ok" ? " dim" : "";
  return (
    <div className={`duel-side ${side}`} title={has ? `n=${v.n} · ${STATUS_LABEL[v.status]}` : undefined}>
      <span className="track">{has && <i className={dim.trim()} style={{ width: `${(100 * Math.abs(v.value!)) / max}%`, background: color }} />}</span>
      <span className={`v${has ? dim : " dim"}`}>{has ? fmtValue(v) : "—"}</span>
    </div>
  );
}

/** Same metric for every player in one row, with a bar per player scaled to the row maximum. */
export function PlayerCompare({ stats, id }: { stats: MatchStats; id?: string }) {
  const [expanded, setExpanded] = useState(false);
  const rows = compareRows(stats.players);
  if (!rows.length) return null;
  const duel = stats.players.length === 2;
  // The backend sends ~45 metrics per player: start with the headline rows, the button shows all of them by group.
  const head = HEAD_PLAYER.map((k) => rows.find((r) => r.key === k)).filter((r): r is CompareRow => !!r);
  const foldable = rows.length > COMPARE_ROWS && head.length >= 3;
  const all = !foldable || expanded;
  const groups: { id: GroupId | null; rows: CompareRow[] }[] = all
    ? foldable
      ? GROUP_ORDER.map((g) => ({ id: g, rows: rows.filter((r) => groupOf(`p.${r.key}`) === g) })).filter((g) => g.rows.length > 0)
      : [{ id: null, rows }]
    : [{ id: null, rows: head }];
  const rowMax = (r: CompareRow) => Math.max(1e-9, ...r.values.map((v) => Math.abs(v?.value ?? 0)));
  const rowNote = (r: CompareRow) => r.values.find((v) => v?.note)?.note ?? undefined;
  return (
    <section id={id} className="card">
      <div className="card-head">
        <h2>双方对比</h2>
        <span className="sub">同一指标并排比较，条长按该行最大值缩放</span>
      </div>
      {duel ? (
        <div className="duel">
          <div className="duel-head">
            {stats.players.map((p, i) => (
              <span key={p.player_id} className="who duel-who" style={{ gridColumn: i === 0 ? 1 : 3 }}>
                <SideBadge id={p.player_id} />
                {labelOf(p.player_id)}
              </span>
            ))}
          </div>
          {groups.map((g) => (
            <Fragment key={g.id ?? "all"}>
              {g.id && <div className="duel-group">{GROUP_LABEL[g.id]}</div>}
              {g.rows.map((r) => (
                <div key={r.key} className="duel-row" title={rowNote(r)}>
                  <Cell v={r.values[0]} max={rowMax(r)} color={colorOf(stats.players[0].player_id)} side="left" />
                  <div className="label">{r.label}</div>
                  <Cell v={r.values[1]} max={rowMax(r)} color={colorOf(stats.players[1].player_id)} side="right" />
                </div>
              ))}
            </Fragment>
          ))}
        </div>
      ) : (
        <div className="table-scroll">
          <table className="compare-table">
            <thead>
              <tr>
                <th>指标</th>
                {stats.players.map((p) => (
                  <th key={p.player_id}>
                    <Who side={p.player_id} />
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {groups.map((g) => (
                <Fragment key={g.id ?? "all"}>
                  {g.id && (
                    <tr className="group-row">
                      <td colSpan={stats.players.length + 1}>{GROUP_LABEL[g.id]}</td>
                    </tr>
                  )}
                  {g.rows.map((r) => (
                    <tr key={r.key} title={rowNote(r)}>
                      <td className="cmp-label">{r.label}</td>
                      {r.values.map((v, i) => (
                        <td key={i}>
                          {v && v.value != null ? (
                            <div className="cmp-cell" title={`n=${v.n} · ${STATUS_LABEL[v.status]}`}>
                              <span className={`mono${v.status === "ok" ? "" : " muted"}`}>{fmtValue(v)}</span>
                              <span className="cmp-bar">
                                <i className={v.status === "ok" ? undefined : "dim"} style={{ width: `${(100 * Math.abs(v.value)) / rowMax(r)}%`, background: colorOf(stats.players[i].player_id) }} />
                              </span>
                            </div>
                          ) : (
                            <span className="muted">—</span>
                          )}
                        </td>
                      ))}
                    </tr>
                  ))}
                </Fragment>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {foldable && <MoreButton open={expanded} onToggle={() => setExpanded(!expanded)} more={`显示全部 ${rows.length} 项对比`} less="只看主要指标" />}
      <div className="card-foot">灰色数值是样本少或待复核的指标，悬停可看样本量。</div>
    </section>
  );
}
