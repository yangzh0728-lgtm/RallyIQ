import { useEffect, useRef, useState } from "react";
import { CourtGlyph, Empty, Icon, Page, PageHeader, StatTile, type IconName } from "../components/ui";
import { api } from "../lib/api";
import { formatMs } from "../lib/corrections";
import { colorOf } from "../lib/players";
import { videoLink } from "../lib/report";
import type { Habit, HabitEvidence, OpponentReport, PersonSummary, PlayerProfile, ProfileMetric, ProfileVideo, RetestResult } from "../lib/types";

const TREND: Record<ProfileMetric["trend"], string> = { up: "↑", down: "↓", flat: "→", new: "新增", insufficient: "数据不足" };
const RETEST: Record<RetestResult["status"], string> = {
  improved: "改善", worse: "变差", changed: "有变化", unchanged: "持平", pending: "等下一场", not_measurable: "无法测量",
};
const RETEST_ICON: Record<RetestResult["status"], IconName> = {
  improved: "check", worse: "alert", changed: "swap", unchanged: "arrowRight", pending: "clock", not_measurable: "info",
};
const CONF: Record<string, string> = { high: "高", medium: "中", low: "低" };
const CONDITION_ZH: Record<string, string> = {
  camera_view: "机位", camera_height: "机位高度", fps: "帧率", resolution: "分辨率", light: "光照", match_format: "单/双打",
};
const VALUE_ZH: Record<string, Record<string, string>> = {
  camera_view: { end_center: "后场正中", end_oblique: "后场偏斜", side: "侧面", unknown: "机位未知" },
  camera_height: { high: "高位", medium: "中等高度", low: "低位", unknown: "高度未知" },
  light: { normal: "光照正常", dark: "光照偏暗", bright: "过曝" },
  match_format: { singles: "单打", doubles: "双打" },
};
const condLabel = (key: string, value: string) => VALUE_ZH[key]?.[value] ?? (key === "fps" ? `${value} fps` : value);

const link = (videoId: string, tMs: number) => `#/videos/${encodeURIComponent(videoId)}?t=${tMs}`;
const reportLink = (videoId: string) => `#/videos/${encodeURIComponent(videoId)}/report`;
const profileLink = (p: PersonSummary) => (p.is_self ? "#/profile" : `#/profile/${encodeURIComponent(p.person)}`);

const valid = (iso: string) => !Number.isNaN(new Date(iso).getTime());
const shortDate = (iso: string, withYear = false) => {
  if (!valid(iso)) return iso;
  const d = new Date(iso);
  return `${withYear ? `${d.getFullYear()}/` : ""}${d.getMonth() + 1}/${d.getDate()}`;
};
const clock = (iso: string) => (valid(iso) ? new Date(iso).toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit", hour12: false }) : "");
const when = (iso: string) => (valid(iso) ? `${shortDate(iso)} ${clock(iso)}` : iso);
const fullDate = (iso: string) => (valid(iso) ? new Date(iso).toLocaleString("zh-CN") : iso);

/** "10/18 – 11/28", with years when the matches are not all from this year. */
function dateRange(videos: ProfileVideo[]): string {
  const times = videos.filter((v) => valid(v.recorded_at)).map((v) => new Date(v.recorded_at).getTime());
  if (times.length === 0) return "";
  const first = new Date(Math.min(...times));
  const last = new Date(Math.max(...times));
  const withYear = first.getFullYear() !== last.getFullYear() || last.getFullYear() !== new Date().getFullYear();
  const a = shortDate(first.toISOString(), withYear);
  const b = shortDate(last.toISOString(), withYear);
  return a === b ? a : `${a} – ${b}`;
}

/** Decimals needed to show every value of one row the same way (at most two, like the old rounding). */
function decimalsOf(values: (number | null | undefined)[]): number {
  let d = 0;
  for (const v of values) {
    if (v == null) continue;
    const s = String(Math.round(v * 100) / 100);
    const dot = s.indexOf(".");
    if (dot >= 0) d = Math.max(d, s.length - dot - 1);
  }
  return Math.min(d, 2);
}
const signed = (v: number, d: number) => `${v > 0 ? "+" : v < 0 ? "−" : ""}${Math.abs(v).toFixed(d)}`;

/** A number with its unit set smaller, or a dash when there is no value. */
function Val({ v, unit, d }: { v: number | null; unit?: string | null; d: number }) {
  // The unit keeps its width when there is no value, so the dash lines up with the digits above and below.
  return (
    <>
      {v == null ? <span className="muted">—</span> : v.toFixed(d)}
      {unit && <span className={`u${v == null ? " ghost" : ""}`}>{unit}</span>}
    </>
  );
}

/** Only win and coverage rates have a clear "better" direction; other metrics just report the change. */
const higherIsBetter = (key: string) => key.startsWith("cover.") || (key.startsWith("win.") && !key.startsWith("win.lost_"));
const trendTone = (m: ProfileMetric): "up" | "down" | "plain" => ((m.trend === "up" || m.trend === "down") && higherIsBetter(m.key) ? m.trend : "plain");

export function Profile({ person }: { person: string | null }) {
  const [people, setPeople] = useState<PersonSummary[] | null>(null);
  const [profile, setProfile] = useState<PlayerProfile | null>(null);
  const [opp, setOpp] = useState<OpponentReport | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.listPeople().then(setPeople, () => setPeople([]));
    setProfile(null);
    setOpp(null);
    setError(null);
    api.getProfile(person).then(setProfile, (e: Error) => setError(e.message));
    if (person) api.getOpponent(person).then(setOpp, () => setOpp(null));
  }, [person]);

  // With many people the switcher scrolls sideways: keep the current one in view.
  const tabsRef = useRef<HTMLElement>(null);
  useEffect(() => {
    tabsRef.current?.querySelector(".on")?.scrollIntoView({ block: "nearest", inline: "nearest" });
  }, [people, person]);

  // The user's own entry first; the rest keep the order the server gave.
  const tabs = people ? [...people].sort((a, b) => Number(b.is_self) - Number(a.is_self)) : [];
  const switcher = tabs.length > 0 && (
    <nav className="seg person-tabs" aria-label="切换档案" ref={tabsRef}>
      {tabs.map((p) => {
        const on = (person ?? null) === (p.is_self ? null : p.person);
        return (
          <a key={p.person} href={profileLink(p)} className={on ? "on" : ""} aria-current={on ? "page" : undefined} title={p.is_self ? "我的档案" : `${p.person} 的档案`}>
            {p.is_self && <Icon name="user" />}
            {p.person}
            <span className="n mono">· {p.videos} 场</span>
          </a>
        );
      })}
    </nav>
  );

  return (
    <Page className="wide profile" header={<PageHeader crumbs={[{ label: "比赛库", href: "#/" }]} title={person ? `${person} 的档案` : "我的档案"} actions={switcher || undefined} />}>
      {error ? (
        <section className="card">
          <Empty icon="user" title="档案加载失败">
            {error}
            {person && (
              <>
                {" "}
                <a href="#/profile">回到我的档案</a>
              </>
            )}
          </Empty>
        </section>
      ) : !profile ? (
        <section className="card">
          <p className="muted">加载中…</p>
        </section>
      ) : profile.videos.length === 0 ? (
        <section className="card">
          <Empty icon="user" title="还没有写入档案的比赛">
            在赛后报告页底部“写入个人档案”里填好谁是你、对手是谁，确认后写入。 <a href="#/">去比赛库选一场</a>
          </Empty>
        </section>
      ) : (
        <ProfileBody profile={profile} opp={opp} people={tabs} />
      )}
    </Page>
  );
}

function ProfileBody({ profile, opp, people }: { profile: PlayerProfile; opp: OpponentReport | null; people: PersonSummary[] }) {
  const { videos, metrics, retests } = profile;
  const up = metrics.filter((m) => m.trend === "up").length;
  const down = metrics.filter((m) => m.trend === "down").length;
  const improved = retests.filter((r) => r.status === "improved").length;
  const pending = retests.filter((r) => r.status === "pending").length;
  return (
    <>
      <div className="stat-strip">
        <StatTile icon="video" label="已记录比赛" value={videos.length} unit="场" sub={dateRange(videos)} />
        <StatTile icon="trend" label="追踪指标" value={metrics.length} unit="项" sub={`${up} 项上升 · ${down} 项下降`} />
        <StatTile
          icon="target"
          label="训练复测"
          value={retests.length}
          unit="项"
          sub={retests.length > 0 ? `${improved} 项改善${pending > 0 ? ` · ${pending} 项等下一场` : ""}` : "还没有可复测的训练建议"}
        />
        <StatTile icon="database" label="档案日志" value={profile.log_events} unit="条" sub="只追加，撤下也留痕" />
      </div>

      {profile.changes.length > 0 && (
        <section className="card">
          <div className="card-head">
            <h2>最近变化</h2>
            <span className="sub">比较最近两场有效数据</span>
          </div>
          <ul className="change-list">
            {profile.changes.map((c) => (
              <ChangeItem key={c} text={c} />
            ))}
          </ul>
        </section>
      )}

      {opp && <OpponentView r={opp} />}

      {retests.length > 0 && <RetestCard retests={retests} videos={videos} />}

      <MetricCard profile={profile} />

      <section className="card">
        <div className="card-head">
          <h2>比赛</h2>
          <span className="sub">{videos.length} 场，最近的在前</span>
        </div>
        <div className="match-list">
          {[...videos].reverse().map((v) => (
            <MatchRow key={v.video_id} v={v} people={people} />
          ))}
        </div>
        <p className="card-foot">档案日志共 {profile.log_events} 条记录（只追加，撤下和重新写入都会留痕）。</p>
      </section>
    </>
  );
}

/** One change sentence; a short "趋势：" style prefix becomes a tag so the list can be scanned. */
function ChangeItem({ text }: { text: string }) {
  const m = /^([^：，。\s]{2,5})：([\s\S]+)$/.exec(text);
  if (!m) return <li>{text}</li>;
  const caution = m[1] === "冲突提醒" || m[1] === "演示数据";
  // The conflict line names the differing conditions by their field names (camera_view, fps).
  const body = m[1] === "冲突提醒" ? m[2].replace(/[a-z_]+/g, (k) => CONDITION_ZH[k] ?? k) : m[2];
  return (
    <li className="tagged">
      <span className={`kind${caution ? " caution" : ""}`}>{m[1]}</span>
      <span>{body}</span>
    </li>
  );
}

// ------------------------------------------------------------------ opponent habits

function EvidenceLink({ e, names, dim }: { e: HabitEvidence; names: Map<string, string>; dim?: boolean }) {
  return (
    <a className={`tc${dim ? " dim" : ""}`} href={link(e.video_id, e.t_ms)} title={`${names.get(e.video_id) ?? e.video_id} · ${formatMs(e.t_ms)}`}>
      {formatMs(e.t_ms)}
    </a>
  );
}

function HabitRow({ h, names }: { h: Habit; names: Map<string, string> }) {
  const pct = Math.round(h.p * 100);
  return (
    <tr>
      <td>{h.condition}</td>
      <td>
        <b className="outcome">{h.outcome}</b>
      </td>
      <td>
        <div className="pbar">
          <div className="bar">
            <i style={{ width: `${Math.max(0, Math.min(100, pct))}%` }} />
          </div>
          <span className="mono">
            <b>{pct}%</b>
            <span className="muted">（{h.hits}/{h.n}）</span>
          </span>
        </div>
      </td>
      <td>
        <span className={`pill conf-${h.confidence}`}>{CONF[h.confidence]}</span>
      </td>
      <td>
        <div className="evidence">
          {h.evidence.map((e) => (
            <EvidenceLink key={e.video_id + e.shot_id} e={e} names={names} />
          ))}
          {h.counter_examples.length > 0 && (
            <span className="evidence counter">
              <span className="muted small">反例</span>
              {h.counter_examples.map((e) => (
                <EvidenceLink key={e.video_id + e.shot_id} e={e} names={names} dim />
              ))}
            </span>
          )}
        </div>
      </td>
    </tr>
  );
}

function OpponentView({ r }: { r: OpponentReport }) {
  const names = new Map(r.videos.map((v) => [v.video_id, v.filename ?? v.video_id]));
  const groups = [
    { title: "发球与击球位置", rows: r.habits },
    { title: "接我方来球后的下一拍", rows: r.next_shot },
  ].filter((g) => g.rows.length > 0);
  return (
    <section className="card">
      <div className="card-head">
        <h2>对手习惯</h2>
        <span className="sub">
          {r.videos.length} 场 · {r.shots} 拍，{r.typed_shots} 拍有类型
        </span>
      </div>
      {r.caveats.length > 0 && (
        <div className="notice habit-caveats">
          <Icon name="info" />
          <ul className="caveats">
            {r.caveats.map((c) => (
              <li key={c}>{c}</li>
            ))}
          </ul>
        </div>
      )}
      {groups.length === 0 ? (
        <p className="muted">证据不足：还没有出现至少 5 次、占比 40% 以上的倾向。</p>
      ) : (
        <div className="table-scroll">
          <table className="habit-table">
            <thead>
              <tr>
                <th>情境</th>
                <th>倾向</th>
                <th>占比</th>
                <th>可信度</th>
                <th>视频证据 / 反例</th>
              </tr>
            </thead>
            {groups.map((g) => (
              <tbody key={g.title}>
                <tr className="group">
                  <th colSpan={5} scope="rowgroup">
                    {g.title}
                  </th>
                </tr>
                {g.rows.map((h) => (
                  <HabitRow key={h.condition + h.outcome} h={h} names={names} />
                ))}
              </tbody>
            ))}
          </table>
        </div>
      )}
    </section>
  );
}

// ------------------------------------------------------------------ training retests

function RetestCard({ retests, videos }: { retests: RetestResult[]; videos: ProfileVideo[] }) {
  const dateOf = (id: string | null) => {
    const v = id ? videos.find((x) => x.video_id === id) : null;
    return v ? shortDate(v.recorded_at) : null;
  };
  return (
    <section className="card">
      <div className="card-head">
        <h2>训练复测</h2>
        <span className="sub">每条训练建议在下一场比赛里用同一指标复测</span>
      </div>
      <div className="table-scroll">
        <table className="retest-table">
          <thead>
            <tr>
              <th>训练</th>
              <th>复测指标</th>
              <th className="num">上次</th>
              <th className="num">这次</th>
              <th>结果</th>
            </tr>
          </thead>
          <tbody>
            {retests.map((r, i) => {
              const d = decimalsOf([r.before, r.after, r.delta]);
              const from = dateOf(r.from_video_id);
              const to = dateOf(r.to_video_id);
              return (
                <tr key={i}>
                  <td title={r.why}>
                    <b className="drill">{r.drill}</b>
                    {r.focus && <div className="small muted">针对：{r.focus}</div>}
                    {r.sources?.map((s) => (
                      <div key={s.note_id} className="small">
                        <a className="source" href={videoLink(s)} target="_blank" rel="noreferrer">
                          <Icon name="video" />
                          {s.coach.split(" (")[0]} · {s.title}
                        </a>
                      </div>
                    ))}
                  </td>
                  <td>
                    {r.label ?? r.metric}
                    {from && (
                      <div className="small muted mono">
                        {from} → {to ?? "下一场"}
                      </div>
                    )}
                  </td>
                  <td className="num mono">
                    <Val v={r.before} unit={r.unit} d={d} />
                  </td>
                  <td className="num mono">
                    <Val v={r.after} unit={r.unit} d={d} />
                  </td>
                  <td>
                    <span className={`result ${r.status}`}>
                      <Icon name={RETEST_ICON[r.status]} />
                      {RETEST[r.status]}
                      {r.delta != null && (
                        <span className="delta mono">
                          {signed(r.delta, d)}
                          {r.unit}
                        </span>
                      )}
                    </span>
                    {r.same_conditions === false && <div className="muted small">拍摄条件不同，仅供参考</div>}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}

// ------------------------------------------------------------------ metric trends

const SPARK_W = 96;
const SPARK_H = 24;
const TONE_COLOR = { up: "var(--ok)", down: "var(--bad)", plain: "var(--text-2)" } as const;

/** Trend line of the measured values in match order; the last point carries the trend colour. */
function Spark({ m }: { m: ProfileMetric }) {
  const pts = m.points.filter((p): p is typeof p & { value: number } => p.value != null);
  if (pts.length < 2) return null;
  const values = pts.map((p) => p.value);
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  const mid = (lo + hi) / 2;
  // Never stretch a small wobble over the full height: the scale spans at least a quarter of the level.
  const span = Math.max(hi - lo, Math.abs(mid) * 0.25, 1e-9);
  const pad = 4;
  const x = (i: number) => pad + (i * (SPARK_W - 2 * pad)) / (pts.length - 1);
  const y = (v: number) => SPARK_H / 2 - ((v - mid) / span) * (SPARK_H - 2 * pad);
  const d = decimalsOf(values);
  const label = pts.map((p) => `${shortDate(p.recorded_at)} ${p.value.toFixed(d)}${m.unit ?? ""}`).join(" → ");
  const last = pts.length - 1;
  return (
    <svg className="spark" width={SPARK_W} height={SPARK_H} viewBox={`0 0 ${SPARK_W} ${SPARK_H}`} role="img" aria-label={`${m.label}：${label}`}>
      <title>{label}</title>
      <polyline points={pts.map((p, i) => `${x(i).toFixed(1)},${y(p.value).toFixed(1)}`).join(" ")} fill="none" stroke="var(--text-2)" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round" />
      <circle cx={x(last)} cy={y(values[last])} r={3} fill={TONE_COLOR[trendTone(m)]} stroke="var(--surface)" strokeWidth={1.5} />
    </svg>
  );
}

function TrendCell({ m, d }: { m: ProfileMetric; d: number }) {
  const moving = m.trend === "up" || m.trend === "down" || m.trend === "flat";
  return (
    <span className="trend-cell">
      <span className={`trend ${m.trend}${(m.trend === "up" || m.trend === "down") && trendTone(m) === "plain" ? " plain" : ""}`}>
        {TREND[m.trend]}
        {moving && m.delta != null && (
          <span className="mono">
            {signed(m.delta, d)}
            {m.unit}
          </span>
        )}
      </span>
      {m.note && <Icon name="alert" className="note-mark" style={{ color: "var(--warn)" }} />}
    </span>
  );
}

function MetricCard({ profile }: { profile: PlayerProfile }) {
  const { videos, metrics } = profile;
  // Month / day is enough while every match is from this year; otherwise the year has to be shown.
  const years = new Set(videos.filter((v) => valid(v.recorded_at)).map((v) => new Date(v.recorded_at).getFullYear()));
  const withYear = years.size > 1 || (years.size === 1 && !years.has(new Date().getFullYear()));
  const dates = videos.map((v) => shortDate(v.recorded_at, withYear));
  // With many matches the table is wider than the card: start at the newest matches and the trend columns.
  const scrollRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollLeft = el.scrollWidth;
  }, [videos.length]);
  return (
    <section className="card">
      <div className="card-head">
        <h2>指标走势</h2>
        <span className="sub">{videos.length} 场</span>
      </div>
      <div className="table-scroll" ref={scrollRef}>
        <table className="metric-table">
          <thead>
            <tr>
              <th>指标</th>
              {videos.map((v, i) => (
                <th key={v.video_id} className="num mono" title={`${fullDate(v.recorded_at)} · ${v.filename ?? v.video_id}`}>
                  {dates[i]}
                  {dates.indexOf(dates[i]) !== dates.lastIndexOf(dates[i]) && <div className="th-time">{clock(v.recorded_at)}</div>}
                </th>
              ))}
              <th>走势</th>
              <th>趋势</th>
              <th>可信度</th>
            </tr>
          </thead>
          <tbody>
            {metrics.map((m) => {
              const d = decimalsOf([...m.points.map((p) => p.value), m.delta]);
              return (
                <tr key={m.key} title={m.note ?? undefined}>
                  <td className="name">
                    {m.label}
                    {m.note && <div className="small muted metric-note">{m.note}</div>}
                  </td>
                  {videos.map((v) => {
                    const p = m.points.find((x) => x.video_id === v.video_id);
                    return (
                      <td key={v.video_id} className={`num mono ${p?.status !== "ok" ? "muted" : ""}`}>
                        {p ? <Val v={p.value} unit={m.unit} d={d} /> : ""}
                      </td>
                    );
                  })}
                  <td>
                    <Spark m={m} />
                  </td>
                  <td>
                    <TrendCell m={m} d={d} />
                  </td>
                  <td>
                    <span className={`pill conf-${m.confidence}`}>{CONF[m.confidence]}</span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="card-foot">
        灰色数值是待复核或不可判定的场次，空格表示那一场没有这项指标。趋势比较最近两场有效数据；只有胜率和覆盖率按好坏着色，其余指标只标方向。
      </p>
    </section>
  );
}

// ------------------------------------------------------------------ matches

function MatchRow({ v, people }: { v: ProfileVideo; people: PersonSummary[] }) {
  const href = reportLink(v.video_id);
  const format = v.conditions.match_format;
  const conditions = Object.entries(v.conditions).filter(([k]) => k !== "match_format");
  return (
    <div className="match-row">
      <a className="video-thumb" href={href} tabIndex={-1} aria-hidden="true">
        <CourtGlyph />
        {format && <span className="fmt">{condLabel("match_format", format)}</span>}
      </a>
      <div className="info">
        <a className="title" href={href} title="打开赛后报告">
          {v.filename ?? v.video_id}
        </a>
        <div className="meta">
          <span className="mono" title={fullDate(v.recorded_at)}>
            {when(v.recorded_at)}
          </span>
          <span className="who">
            {v.player_id && (
              <>
                <i className="dot" style={{ background: colorOf(v.player_id) }} />
                {v.player_id} ·{" "}
              </>
            )}
            对手{" "}
            {v.opponents.length === 0
              ? "—"
              : v.opponents.map((name, i) => {
                  const p = people.find((x) => x.person === name);
                  return (
                    <span key={name}>
                      {i > 0 && "、"}
                      {p ? <a href={profileLink(p)}>{name}</a> : name}
                    </span>
                  );
                })}
          </span>
        </div>
      </div>
      <div className="conds">
        {conditions.map(([k, value]) => (
          <span key={k} className="chip" title={`${CONDITION_ZH[k] ?? k}：${value}`}>
            {condLabel(k, value)}
          </span>
        ))}
      </div>
      <a className="btn small" href={href}>
        赛后报告
        <Icon name="arrowRight" />
      </a>
    </div>
  );
}
