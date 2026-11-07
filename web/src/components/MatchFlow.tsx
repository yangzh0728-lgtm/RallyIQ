import { useMemo, useState } from "react";
import { compareRows, leadSeries } from "../lib/flow";
import { colorOf } from "../lib/players";
import { fmtValue, STATUS_LABEL } from "../lib/report";
import { formatMs } from "../lib/corrections";
import type { GameScore, MatchStats, Metric, RallyLine } from "../lib/types";

const sideColor = (s: string | null | undefined) => (s ? colorOf(s) : "var(--line)");
const at = (videoId: string, ms: number) => `#/videos/${encodeURIComponent(videoId)}?t=${ms}`;

function Who({ side }: { side: string | null }) {
  if (!side) return <span className="muted">—</span>;
  return (
    <span className="row" style={{ gap: 4, display: "inline-flex" }}>
      <span className="dot" style={{ background: sideColor(side) }} />
      {side}
    </span>
  );
}

/** Games, lead after every decided rally, and the flow metrics (runs, lead changes). */
export function ScoreFlow({ stats }: { stats: MatchStats }) {
  const rallies = stats.rallies ?? [];
  const games = stats.games ?? [];
  const decided = rallies.find((r) => r.score);
  if (!decided) {
    return (
      <section className="card">
        <h2>比分走势</h2>
        <p className="muted small">
          还没有标注回合胜负。在工作台的修正面板里给回合设置胜者后，这里会按 21 分制推算每局比分、领先变化和连续得分。
        </p>
      </section>
    );
  }
  const sides = Object.keys(decided.score!) as [string, string];
  const series = leadSeries(rallies, sides);
  const top = Math.max(3, ...series.map((s) => Math.abs(s.lead)));
  const padL = 28;
  const W = 720;
  const H = 140;
  const mid = H / 2;
  const half = H / 2 - 8;
  // Bars at most 22 units wide, so a short match does not turn into two giant blocks.
  const bw = Math.min((W - padL) / series.length, 22);
  const gap = Math.min(2, bw * 0.25);
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
  return (
    <section className="card">
      <h2>比分走势</h2>
      <div className="flow-games">
        {games.map((g: GameScore) => (
          <div key={g.game_no} className="flow-game">
            <div className="mono value">
              {g.score[sides[0]] ?? 0} : {g.score[sides[1]] ?? 0}
            </div>
            <div className="small muted">
              第 {g.game_no} 局 · {g.winner ? <Who side={g.winner} /> : "未完成"}
              {g.winner ? " 胜" : ""}
            </div>
          </div>
        ))}
      </div>
      <div className="row small" style={{ margin: "8px 0 4px" }}>
        <span className="muted">领先分差（向上 {sides[0]} 领先，向下 {sides[1]} 领先；竖虚线是换局）</span>
        <span className="spacer" />
        {sides.map((s) => (
          <Who key={s} side={s} />
        ))}
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="flow-chart" role="img" aria-label="比分走势">
        <line x1={padL} x2={W} y1={mid} y2={mid} stroke="var(--line)" />
        <text x={padL - 4} y={14} className="axis" textAnchor="end">+{top}</text>
        <text x={padL - 4} y={mid + 4} className="axis" textAnchor="end">0</text>
        <text x={padL - 4} y={H - 4} className="axis" textAnchor="end">-{top}</text>
        {series.map(({ rally, lead }, i) => {
          const x = padL + i * bw;
          const h = Math.max((half * Math.abs(lead)) / top, 1);
          const newGame = i > 0 && series[i - 1].rally.game_no !== rally.game_no;
          const sc = `${rally.score![sides[0]]}-${rally.score![sides[1]]}`;
          return (
            <g key={rally.rally_id}>
              {newGame && <line x1={x} x2={x} y1={4} y2={H - 4} stroke="var(--muted)" strokeDasharray="3 3" />}
              <rect
                x={x + gap / 2}
                y={lead > 0 ? mid - h : mid}
                width={Math.max(bw - gap, 1)}
                height={h}
                rx={Math.min(2, (bw - gap) / 2)}
                fill={lead > 0 ? sideColor(sides[0]) : lead < 0 ? sideColor(sides[1]) : "var(--muted)"}
              >
                <title>{`第 ${rally.rally_no} 回合 · ${formatMs(rally.start_ms)} · ${rally.winner} 得分 · 比分 ${sc}`}</title>
              </rect>
            </g>
          );
        })}
      </svg>
      <div className="small muted">
        按已标注的回合胜负推算（21 分制，领先 2 分获胜，30 分封顶）。{facts.join("；")}
        {facts.length ? "。" : ""}
        {undecided > 0 && `还有 ${undecided} 个回合没有标注胜负，比分可能偏低。`}
      </div>
    </section>
  );
}

type Filter = "all" | "long" | "short" | string;

/** One row per rally: time (jumps to the video), shots, duration, server, winner, score, last shot, tempo. */
export function RallyTable({ stats, videoId }: { stats: MatchStats; videoId: string }) {
  const rallies = stats.rallies ?? [];
  const [filter, setFilter] = useState<Filter>("all");
  const sides = useMemo(() => [...new Set(rallies.map((r) => r.winner).filter((w): w is NonNullable<typeof w> => !!w))].sort(), [rallies]);
  if (!rallies.length) return null;
  const maxShots = Math.max(1, ...rallies.map((r) => r.shots));
  const shown = rallies.filter((r: RallyLine) =>
    filter === "all" ? true : filter === "long" ? r.shots >= 7 : filter === "short" ? r.shots <= 4 : r.winner === filter,
  );
  const half = Math.floor(rallies.length / 2);
  const mean = (xs: number[]) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null);
  const firstLen = mean(rallies.slice(0, half).map((r) => r.shots));
  const secondLen = mean(rallies.slice(half).map((r) => r.shots));
  return (
    <section className="card">
      <div className="row" style={{ marginBottom: 8 }}>
        <h2 style={{ margin: 0 }}>逐回合</h2>
        <span className="muted small">点时间跳回视频；柱长是拍数，颜色是得分方</span>
        <span className="spacer" />
        <select value={filter} onChange={(e) => setFilter(e.target.value)} aria-label="筛选回合">
          <option value="all">全部 {rallies.length} 个回合</option>
          <option value="long">长回合（≥7 拍）</option>
          <option value="short">短回合（≤4 拍）</option>
          {sides.map((s) => (
            <option key={s} value={s}>
              {s} 得分的回合
            </option>
          ))}
        </select>
      </div>
      {firstLen != null && secondLen != null && half >= 3 && (
        <div className="small muted" style={{ marginBottom: 6 }}>
          前半段平均每回合 {firstLen.toFixed(1)} 拍，后半段 {secondLen.toFixed(1)} 拍
          {Math.abs(secondLen - firstLen) >= 1 ? (secondLen > firstLen ? "：后半段相持更多" : "：后半段回合变短") : "：前后节奏接近"}。
        </div>
      )}
      <div className="table-scroll">
        <table className="rally-table">
          <thead>
            <tr>
              <th>#</th>
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
            {shown.map((r) => (
              <tr key={r.rally_id}>
                <td className="mono muted">{r.rally_no}</td>
                <td>
                  <a className="mono" href={at(videoId, r.start_ms)} title="在工作台打开这个回合">
                    {formatMs(r.start_ms)}
                  </a>
                </td>
                <td>
                  <div className="rally-len">
                    <i style={{ width: `${(100 * r.shots) / maxShots}%`, background: sideColor(r.winner) }} />
                    <span className="mono">{r.shots}</span>
                  </div>
                </td>
                <td className="mono num">{r.duration_s.toFixed(1)} s</td>
                <td>
                  <Who side={r.server} />
                </td>
                <td>
                  <Who side={r.winner} />
                </td>
                <td className="mono num">{r.score ? Object.values(r.score).join("-") : "—"}</td>
                <td className="small">{r.last_hitter || r.last_shot_type ? `${r.last_hitter ?? "?"} ${r.last_shot_type ?? ""}` : "—"}</td>
                <td className="mono num">{r.mean_interval_s != null ? `${r.mean_interval_s.toFixed(2)} s` : "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {!shown.length && <p className="muted small">没有符合条件的回合。</p>}
    </section>
  );
}

/** Same metric for every player in one row, with a bar per player scaled to the row maximum. */
export function PlayerCompare({ stats }: { stats: MatchStats }) {
  const rows = compareRows(stats.players);
  if (!rows.length) return null;
  return (
    <section className="card">
      <h2>双方对比</h2>
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
            {rows.map((r) => {
              const max = Math.max(1e-9, ...r.values.map((v) => Math.abs(v?.value ?? 0)));
              return (
                <tr key={r.key} title={r.values.find((v) => v?.note)?.note ?? undefined}>
                  <td className="small">{r.label}</td>
                  {r.values.map((v, i) => (
                    <td key={i}>
                      {v && v.value != null ? (
                        <div className="cmp-cell" title={`n=${v.n} · ${STATUS_LABEL[v.status]}`}>
                          <span className={`mono${v.status === "ok" ? "" : " muted"}`}>{fmtValue(v)}</span>
                          <span className="cmp-bar">
                            <i style={{ width: `${(100 * Math.abs(v.value)) / max}%`, background: colorOf(stats.players[i].player_id) }} />
                          </span>
                        </div>
                      ) : (
                        <span className="muted">—</span>
                      )}
                    </td>
                  ))}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div className="small muted">灰色数值是样本少或待复核的指标，悬停可看样本量。</div>
    </section>
  );
}
