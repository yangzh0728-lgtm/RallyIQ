import { Fragment, useCallback, useEffect, useMemo, useRef, useState, type CSSProperties } from "react";
import { GROUP_LABEL, GROUP_ORDER, groupOf, HEAD_PLAYER, MoreButton, PlayerCompare, RallyTable, ScoreFlow, type GroupId } from "../components/MatchFlow";
import { MemoryCommit } from "../components/MemoryCommit";
import { Empty, Icon, Page, PageHeader, SideBadge, VideoTabs } from "../components/ui";
import { api } from "../lib/api";
import { download } from "../lib/export";
import { compareRows } from "../lib/flow";
import { colorOf, labelOf } from "../lib/players";
import { clock, CONF_LABEL, evidenceText, fmtValue, indexStats, isMetric, KIND_LABEL, PHASE_LABEL, reportMarkdown, shotZh, STATUS_LABEL, styleWho, videoLink, type Evidence } from "../lib/report";
import { TEAM_MEMBERS, TEAM_OF, type CoachReport, type Distribution, type GameScore, type MatchReport, type MatchStats, type Metric, type PlayerStyle, type Status, type TeamId, type Video } from "../lib/types";

const STATUS_HELP = "状态按样本量和球场标定情况给出：可用 / 待复核 / 不可判定。统计只使用修正后的事件，原始模型预测不会被改写。";
const STATUSES: Status[] = ["ok", "needs_review", "undeterminable"];

const isTeam = (id: string | null | undefined): id is TeamId => id === "A" || id === "B";
const sideName = (id: string) => (isTeam(id) ? `${id} 队` : id);
const subjectOf = (key: string) => key.slice(0, Math.max(0, key.indexOf(".")));
const suffixOf = (key: string) => key.slice(key.indexOf(".") + 1);

// The numbers shown first; everything else stays one click away (the backend sends 18 match metrics,
// ~45 per player and, in doubles, ~45 per team).
const HEAD_MATCH = ["match.rallies", "match.shots", "match.rally_len.mean", "match.rally_len.max", "match.rally_dur.mean_s", "match.confirmed_shots_pct", "match.winners_known"];
const HEAD_PLAYER_DISTS = ["pos.depth_zones", "shot_types"];
const HEAD_TEAM = ["dbl.formation.fb_pct", "dbl.formation.sbs_pct", "dbl.rotation.switches_per_rally", "dbl.gap.mean_m", "dbl.cover.ratio_pct", "style.attack_pct"];
const HEAD_TEAM_DISTS = ["style.phases", "dbl.who.smash", "dbl.who.net"];
const HEAD_MAX = 8;
const ADVICE_ROWS = 6;

type DataTab = "overview" | "all" | GroupId;

interface Subject {
  id: string;
  name: string;
  sub?: string;
  metrics: Metric[];
  distributions: Distribution[];
}

interface SubjectPlan {
  subject: Subject;
  headMetrics: Metric[];
  headDists: Distribution[];
  groups: { id: GroupId; metrics: Metric[]; dists: Distribution[] }[];
  /** There is more than the overview shows, so the group tabs are worth having. */
  extra: boolean;
}

function planSubject(subject: Subject, head: string[], headDists: string[]): SubjectPlan {
  const pick = <T extends { key: string }>(all: T[], keys: string[], fallback: number): T[] => {
    const found = keys.map((k) => all.find((x) => suffixOf(x.key) === k)).filter((x): x is T => !!x);
    return found.length ? found : all.slice(0, fallback);
  };
  const headMetrics = pick(subject.metrics, head, HEAD_MAX);
  const dists = pick(subject.distributions, headDists, 2);
  const groups = GROUP_ORDER.map((id) => ({
    id,
    metrics: subject.metrics.filter((m) => groupOf(m.key) === id),
    dists: subject.distributions.filter((d) => groupOf(d.key) === id),
  })).filter((g) => g.metrics.length + g.dists.length > 0);
  return { subject, headMetrics, headDists: dists, groups, extra: headMetrics.length < subject.metrics.length || dists.length < subject.distributions.length };
}

/** The tab a key can be seen in: the current one if it already shows it, else the key's own group. */
function tabFor(plans: SubjectPlan[], tab: DataTab, key: string): DataTab {
  if (!plans.some((p) => p.extra)) return tab;
  const inHead = plans.some((p) => p.headMetrics.some((m) => m.key === key) || p.headDists.some((d) => d.key === key));
  const group = groupOf(key);
  if (tab === "all" || tab === group || (tab === "overview" && inHead)) return tab;
  if (inHead) return "overview";
  return plans.some((p) => p.groups.some((g) => g.id === group)) ? group : "all";
}

/** The two sides of the match, from the same source the score flow uses (keys of the first decided rally's score). */
function matchSides(stats: MatchStats): { sides: [string, string]; doubles: boolean } {
  const rallies = stats.rallies ?? [];
  const keys = Object.keys(rallies.find((r) => r.score)?.score ?? {});
  const doubles =
    stats.players.length > 2 ||
    keys.some(isTeam) ||
    rallies.some((r) => isTeam(r.server) || isTeam(r.winner)) ||
    (stats.styles ?? []).some((s) => s.kind === "doubles");
  if (keys.length === 2) return { sides: [keys[0], keys[1]], doubles };
  return { sides: doubles ? ["A", "B"] : ["P1", "P2"], doubles };
}

interface Board {
  a: number;
  b: number;
  /** Index of the side whose number is dimmed; only set once the game / match is decided. */
  loser: 0 | 1 | null;
  line: string;
  games: GameScore[];
}

/** What the scoreboard shows: one game's points, or games won when there are several. */
function scoreBoard(stats: MatchStats, sides: [string, string]): Board | null {
  const games = stats.games ?? [];
  const pts = (score: Record<string, number>, s: string) => score[s] ?? 0;
  const lower = (a: number, b: number): 0 | 1 | null => (a < b ? 0 : b < a ? 1 : null);
  if (games.length > 1) {
    const won = sides.map((s) => games.filter((g) => g.winner === s).length);
    const last = games[games.length - 1];
    const decided = !!last.winner && Math.max(...won) >= 2 && won[0] !== won[1];
    const line = decided ? `${sideName(sides[won[0] > won[1] ? 0 : 1])} 胜` : last.winner ? "未分胜负" : `第 ${last.game_no} 局未完成`;
    return { a: won[0], b: won[1], loser: decided ? lower(won[0], won[1]) : null, line: `局分 · ${line}`, games };
  }
  if (games.length === 1) {
    const g = games[0];
    const a = pts(g.score, sides[0]);
    const b = pts(g.score, sides[1]);
    return { a, b, loser: g.winner ? lower(a, b) : null, line: `第 ${g.game_no} 局 · ${g.winner ? `${sideName(g.winner)} 胜` : "未完成"}`, games: [] };
  }
  // Reports from older versions carry rally scores but no games: show the latest running score.
  const last = [...(stats.rallies ?? [])].reverse().find((r) => r.score);
  if (!last?.score) return null;
  return { a: pts(last.score, sides[0]), b: pts(last.score, sides[1]), loser: null, line: `${last.game_no ? `第 ${last.game_no} 局 · ` : ""}未完成`, games: [] };
}

/** Scroll-spy for the section tabs: the active tab follows the scroll position inside .page-scroll. */
function useSectionNav(ids: string[], ready: boolean) {
  const [active, setActive] = useState(ids[0]);
  const lockUntil = useRef(0);
  const clicked = useRef<string | null>(null);
  const key = ids.join("|");
  useEffect(() => {
    if (!ready) return;
    const list = key.split("|");
    const scroller = document.getElementById(list[0])?.closest<HTMLElement>(".page-scroll");
    if (!scroller) return;
    let raf = 0;
    let timer = 0;
    const update = () => {
      raf = 0;
      const wait = lockUntil.current - Date.now();
      if (wait > 0) {
        // A tab click is still scrolling: keep the clicked tab, look again when the scroll has settled.
        window.clearTimeout(timer);
        timer = window.setTimeout(update, wait + 30);
        return;
      }
      const box = scroller.getBoundingClientRect();
      const rect = (id: string) => document.getElementById(id)?.getBoundingClientRect();
      let cur = list[0];
      for (const id of list) {
        const r = rect(id);
        if (r && r.top - box.top <= 96) cur = id;
      }
      const atBottom = scroller.scrollTop > 4 && scroller.scrollTop + scroller.clientHeight >= scroller.scrollHeight - 4;
      if (atBottom) {
        // The last sections can never reach the top: prefer the tab that was clicked, else the last one in view.
        const inView = list.filter((id) => {
          const r = rect(id);
          return !!r && r.top < box.bottom - 24 && list.indexOf(id) >= list.indexOf(cur);
        });
        cur = clicked.current && inView.includes(clicked.current) ? clicked.current : (inView[inView.length - 1] ?? cur);
      } else {
        clicked.current = null;
      }
      setActive(cur);
    };
    const onScroll = () => {
      if (!raf) raf = requestAnimationFrame(update);
    };
    scroller.addEventListener("scroll", onScroll, { passive: true });
    update();
    return () => {
      scroller.removeEventListener("scroll", onScroll);
      if (raf) cancelAnimationFrame(raf);
      window.clearTimeout(timer);
    };
  }, [ready, key]);
  const go = (id: string) => {
    setActive(id);
    clicked.current = id;
    lockUntil.current = Date.now() + 800;
    document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" });
  };
  return [active, go] as const;
}

export function Report({ videoId }: { videoId: string }) {
  const [video, setVideo] = useState<Video | null>(null);
  const [report, setReport] = useState<MatchReport | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [generating, setGenerating] = useState(false);
  const [coachError, setCoachError] = useState<string | null>(null);
  const [focus, setFocus] = useState<{ key: string; n: number } | null>(null);
  const [heroOpen, setHeroOpen] = useState(false);
  const [playerTab, setPlayerTab] = useState<DataTab>("overview");
  const [teamTab, setTeamTab] = useState<DataTab>("overview");

  const load = useCallback(() => {
    Promise.all([api.getVideo(videoId), api.getReport(videoId)]).then(
      ([v, r]) => {
        setVideo(v);
        setReport(r);
      },
      (e: Error) => setError(e.message),
    );
  }, [videoId]);
  useEffect(load, [load]);

  const idx = useMemo(() => (report ? indexStats(report.stats) : new Map<string, Evidence>()), [report]);

  // Match-level numbers stay in the overview; in doubles the per-team ones (keys "A.…" / "B.…") get their own cards.
  const layout = useMemo(() => {
    const stats = report?.stats;
    const metrics = stats?.metrics ?? [];
    const dists = stats?.distributions ?? [];
    const matchMetrics = metrics.filter((m) => !isTeam(subjectOf(m.key)));
    const matchDists = dists.filter((d) => !isTeam(subjectOf(d.key)));
    const found = HEAD_MATCH.map((k) => matchMetrics.find((m) => m.key === k)).filter((m): m is Metric => !!m);
    const heroHead = matchMetrics.length <= HEAD_MAX ? matchMetrics : found.length >= 3 ? found : matchMetrics.slice(0, HEAD_MAX);
    const heroRest = matchMetrics.filter((m) => !heroHead.includes(m));
    const teams = (["A", "B"] as TeamId[])
      .map((t): Subject => ({
        id: t,
        name: labelOf(t),
        metrics: metrics.filter((m) => subjectOf(m.key) === t),
        distributions: dists.filter((d) => subjectOf(d.key) === t),
      }))
      .filter((t) => t.metrics.length + t.distributions.length > 0);
    const fourPlayers = (stats?.players.length ?? 0) > 2;
    const players = (stats?.players ?? []).map(
      (p): Subject => ({
        id: p.player_id,
        name: labelOf(p.player_id),
        sub: fourPlayers && TEAM_OF[p.player_id] ? `${TEAM_OF[p.player_id]} 队` : undefined,
        metrics: p.metrics,
        distributions: p.distributions,
      }),
    );
    return {
      heroHead,
      heroRest,
      matchDists,
      teamPlans: teams.map((t) => planSubject(t, HEAD_TEAM, HEAD_TEAM_DISTS)),
      playerPlans: players.map((p) => planSubject(p, HEAD_PLAYER, HEAD_PLAYER_DISTS)),
    };
  }, [report]);

  const hasStyles = (report?.stats.styles?.length ?? 0) > 0;
  const hasTeams = layout.teamPlans.length > 0;
  const hasPlayers = layout.playerPlans.length > 0;
  const hasCompare = !!report && compareRows(report.stats.players).length > 0;
  const hasRallies = (report?.stats.rallies?.length ?? 0) > 0;
  const tabs = useMemo(
    () =>
      [
        { id: "sec-overview", label: "概览", show: true },
        { id: "sec-coach", label: "AI 点评", show: true },
        { id: "sec-style", label: "打法风格", show: hasStyles },
        { id: "sec-flow", label: "比分走势", show: true },
        { id: "sec-compare", label: "双方对比", show: hasCompare },
        { id: "sec-rallies", label: "逐回合", show: hasRallies },
        { id: "sec-teams", label: "队伍数据", show: hasTeams },
        { id: "sec-players", label: "球员数据", show: hasPlayers },
        { id: "sec-profile", label: "写入档案", show: true },
      ].filter((t) => t.show),
    [hasStyles, hasCompare, hasRallies, hasTeams, hasPlayers],
  );
  const [activeTab, goTab] = useSectionNav(
    tabs.map((t) => t.id),
    !!report && !error,
  );

  const generate = async () => {
    setGenerating(true);
    setCoachError(null);
    try {
      const coach = await api.generateCoach(videoId);
      setReport((r) => (r ? { ...r, coach, coach_stale: coach.stats_fingerprint !== r.stats.fingerprint } : r));
    } catch (e) {
      setCoachError((e as Error).message);
    } finally {
      setGenerating(false);
    }
  };

  const doExport = (format: "md" | "json" | "html") => {
    const url = api.reportExportUrl(videoId, format);
    if (url) {
      window.location.href = url;
      return;
    }
    if (!report || format === "html") return;
    if (format === "json") download(`${videoId}_report.json`, JSON.stringify(report, null, 1), "application/json");
    else download(`${videoId}_report.md`, reportMarkdown(`赛后报告 · ${video?.filename ?? videoId}`, report.stats, report.coach, report.coach_stale), "text/markdown");
  };

  // Jump to a metric / distribution tile: reveal it first if it sits behind "more" or another data tab,
  // then scroll once that render has landed.
  const jump = (key: string) => {
    if (idx.has(key)) {
      const subject = subjectOf(key);
      if (layout.heroRest.some((m) => m.key === key)) setHeroOpen(true);
      else if (layout.teamPlans.some((p) => p.subject.id === subject)) setTeamTab((t) => tabFor(layout.teamPlans, t, key));
      else if (layout.playerPlans.some((p) => p.subject.id === subject)) setPlayerTab((t) => tabFor(layout.playerPlans, t, key));
    }
    setFocus((f) => ({ key, n: (f?.n ?? 0) + 1 }));
  };
  useEffect(() => {
    if (focus) document.getElementById(`m-${focus.key}`)?.scrollIntoView({ behavior: "smooth", block: "center" });
  }, [focus]);
  const focusKey = focus?.key ?? null;

  const ready = !!report && !error;
  const header = (
    <PageHeader
      crumbs={[{ label: "比赛库", href: "#/" }]}
      title={video?.filename ?? "赛后报告"}
      center={<VideoTabs videoId={videoId} active="report" />}
      actions={
        ready ? (
          <>
            <span className="export-label">
              <Icon name="download" />
              导出
            </span>
            <button className="small" onClick={() => doExport("md")} title="导出 Markdown" aria-label="导出 Markdown">
              Markdown
            </button>
            <button className="small" onClick={() => doExport("json")} title="导出 JSON" aria-label="导出 JSON">
              JSON
            </button>
            {api.reportExportUrl(videoId, "html") && (
              <button
                className="small"
                onClick={() => doExport("html")}
                aria-label="导出网页报告"
                title="一个独立的网页文件：图表都在文件里，离线可看，可直接发给教练或球友"
              >
                网页报告
              </button>
            )}
          </>
        ) : undefined
      }
    />
  );

  if (error) {
    return (
      <Page header={header}>
        <section className="card">
          <Empty icon="alert" title="无法生成报告">
            {error}
            <br />
            需要先完成分析（回合与击球事件）。<a href={`#/videos/${encodeURIComponent(videoId)}`}>回到工作台</a>
          </Empty>
        </section>
      </Page>
    );
  }
  if (!report) {
    return (
      <Page header={header}>
        <section className="card">
          <Empty icon="report" title="正在加载报告…">
            正在汇总回合、击球和球员数据。
          </Empty>
        </section>
      </Page>
    );
  }
  const { stats, coach } = report;
  const { sides, doubles } = matchSides(stats);
  const caveats = coach?.caveats ?? stats.caveats;
  const num = (key: string) => {
    const e = idx.get(key);
    return e && isMetric(e) ? e.value : null;
  };
  const nRallies = num("match.rallies") ?? stats.rallies?.length ?? null;
  const nShots = num("match.shots");
  const calib = stats.calibration_status;
  const { heroHead, heroRest, matchDists } = layout;
  const oneDist = matchDists.length === 1 ? matchDists[0] : null;
  const [distTitle, distUnit] = oneDist ? splitLabel(oneDist.label) : ["全场分布", null];
  const restGroups = GROUP_ORDER.map((g) => ({ id: g, metrics: heroRest.filter((m) => groupOf(m.key) === g) })).filter((g) => g.metrics.length > 0);

  return (
    <Page
      className="report"
      header={header}
      subnav={
        <nav className="subnav" aria-label="报告章节">
          {tabs.map((t) => (
            <button key={t.id} className={activeTab === t.id ? "on" : ""} aria-current={activeTab === t.id ? "true" : undefined} onClick={() => goTab(t.id)}>
              {t.label}
            </button>
          ))}
        </nav>
      }
    >
      <section id="sec-overview" className="card hero">
        <div className="hero-top">
          <span className="chip">{doubles ? "双打" : "单打"}</span>
          {stats.duration_ms != null && (
            <span className="hero-fact">
              <Icon name="clock" />
              时长 <b className="mono">{clock(stats.duration_ms)}</b>
            </span>
          )}
          {nRallies != null && (
            <span className="hero-fact">
              <b className="mono">{nRallies}</b> 回合
            </span>
          )}
          {nShots != null && (
            <span className="hero-fact">
              <b className="mono">{nShots}</b> 击球
            </span>
          )}
          <span className={`pill ${calib ?? "undeterminable"}`}>球场标定 · {calib ? STATUS_LABEL[calib] : "未知"}</span>
          <span className="spacer" />
          <span className="fingerprint mono" title={`数据指纹：统计所依据的事件版本。${STATUS_HELP}`}>
            指纹 {stats.fingerprint}
          </span>
        </div>
        <Scoreboard stats={stats} sides={sides} idx={idx} videoId={videoId} />
        <div className="hero-metrics">
          <MetricGrid metrics={heroHead} focus={focusKey} min={150} />
          {heroRest.length > 0 && (
            <>
              {heroOpen &&
                restGroups.map((g) => (
                  <div key={g.id} className="metric-group">
                    <h3 className="group-head">{GROUP_LABEL[g.id]}</h3>
                    <MetricGrid metrics={g.metrics} focus={focusKey} min={150} />
                  </div>
                ))}
              <MoreButton open={heroOpen} onToggle={() => setHeroOpen(!heroOpen)} more={`其余 ${heroRest.length} 项全场指标`} less="收起其余指标" />
            </>
          )}
        </div>
      </section>

      <div className="grid-3-2">
        <section className="card">
          <div className="card-head">
            <h2>{distTitle}</h2>
            {oneDist && (
              <span className="sub mono">
                {distUnit ? `按${distUnit}统计 · ` : ""}n={oneDist.n}
              </span>
            )}
            <span className="spacer" />
            {oneDist && <span className={`pill ${oneDist.status}`}>{STATUS_LABEL[oneDist.status]}</span>}
          </div>
          {matchDists.length > 0 ? (
            matchDists.map((d) => <DistBars key={d.key} d={d} focus={focusKey} bare={!!oneDist} />)
          ) : (
            <p className="muted small">这场比赛没有全场分布数据。</p>
          )}
        </section>
        <section className="card">
          <div className="card-head">
            <Icon name="info" className="head-icon" />
            <h2>数据说明</h2>
          </div>
          {caveats.length > 0 && (
            <ul className="caveats">
              {caveats.map((c) => (
                <li key={c}>{c}</li>
              ))}
            </ul>
          )}
          <div className={`caveat-foot${caveats.length ? "" : " first"}`}>
            <div className="status-legend">
              {STATUSES.map((s) => (
                <span key={s}>
                  <span className={`status-dot ${s}`} />
                  {STATUS_LABEL[s]}
                </span>
              ))}
            </div>
            <p>状态按样本量和球场标定情况给出。统计只使用修正后的事件，原始模型预测不会被改写。</p>
          </div>
        </section>
      </div>

      <section id="sec-coach" className="card coach">
        <div className="card-head">
          <Icon name="sparkles" className="head-icon" />
          <h2>AI 教练点评</h2>
          {coach && (
            <span className="sub">
              {coach.provider}
              {coach.model ? ` · ${coach.model}` : ""} · {new Date(coach.created_at).toLocaleString()}
            </span>
          )}
          <span className="spacer" />
          <button className="primary" disabled={generating} onClick={() => void generate()}>
            {!generating && <Icon name={coach ? "refresh" : "sparkles"} />}
            {generating ? "生成中…（本地模型可能需要几分钟）" : coach ? "重新生成" : "生成 AI 点评"}
          </button>
        </div>
        {coachError && (
          <div className="notice bad">
            <Icon name="alert" />
            <span>生成失败：{coachError}</span>
          </div>
        )}
        {report.coach_stale && (
          <div className="notice warn">
            <Icon name="alert" />
            <span>这份点评生成后数据又有修正（例如改了击球者或胜负），内容可能已过时，建议重新生成。</span>
          </div>
        )}
        {coach?.fallback_reason && (
          <div className="notice">
            <Icon name="info" />
            <span>
              {coach.provider === "offline" ? "AI 模型不可用，改用离线规则模板" : `已改用 ${coach.provider}`}：{coach.fallback_reason}
            </span>
          </div>
        )}
        {coach ? (
          <CoachView coach={coach} idx={idx} onJump={jump} />
        ) : (
          <div className="coach-empty">
            <span className="glyph">
              <Icon name="sparkles" className="lg" />
            </span>
            <div>
              <b>还没有生成点评</b>
              <p>点评会严格依据本页的统计数据生成，每条观点都标明引用的指标和可信度；数据不足时会直接说“证据不足”。</p>
            </div>
          </div>
        )}
      </section>

      {hasStyles && (
        <section id="sec-style" className="card">
          <div className="card-head">
            <h2>打法风格</h2>
            <span className="sub">规则与打法原型匹配，吻合度 0–100</span>
          </div>
          <p className="section-intro">
            按击球类型、攻防状态、节奏和双方站位，用规则和打法原型做匹配（没有用标注数据训练）。吻合度 0–100，参考区间来自职业单打统计，业余比赛看倾向即可。
          </p>
          <div className="report-players">
            {stats.styles!.map((st) => (
              <StyleCard key={st.subject} st={st} videoId={videoId} idx={idx} onJump={jump} />
            ))}
          </div>
        </section>
      )}

      <ScoreFlow id="sec-flow" stats={stats} />
      <PlayerCompare id="sec-compare" stats={stats} />
      <RallyTable id="sec-rallies" stats={stats} videoId={videoId} />

      {hasTeams && (
        <SubjectSection id="sec-teams" title="队伍数据" hint="两人的站位、轮转、分工和打法特征" plans={layout.teamPlans} tab={teamTab} onTab={setTeamTab} focus={focusKey} />
      )}
      {hasPlayers && (
        <SubjectSection id="sec-players" title="球员数据" hint="每位球员的指标与分布" plans={layout.playerPlans} tab={playerTab} onTab={setPlayerTab} focus={focusKey} />
      )}

      <div id="sec-profile">
        <MemoryCommit videoId={videoId} playerIds={stats.players.map((p) => p.player_id)} />
      </div>
    </Page>
  );
}

/** "回合长度分布（回合数）" -> ["回合长度分布", "回合数"]. */
function splitLabel(label: string): [string, string | null] {
  const m = label.match(/^(.+?)（([^（）]+)）$/);
  return m ? [m[1], m[2]] : [label, null];
}

function Scoreboard({ stats, sides, idx, videoId }: { stats: MatchStats; sides: [string, string]; idx: Map<string, Evidence>; videoId: string }) {
  const board = scoreBoard(stats, sides);
  const metricText = (key: string) => {
    const e = idx.get(key);
    return e && isMetric(e) && e.value !== null ? fmtValue(e) : null;
  };
  const side = (id: string, right: boolean) => {
    const rate = metricText(`${id}.win.rate_pct`);
    const shots = metricText(`${id}.shots`);
    return (
      <div className={`sb-side${right ? " right" : ""}`}>
        <SideBadge id={id} />
        <div className="sb-text">
          <div className="name">{labelOf(id)}</div>
          <div className="sub">
            {isTeam(id) ? (
              TEAM_MEMBERS[id].map((pid) => {
                const n = metricText(`${pid}.shots`);
                return (
                  <span key={pid} className="who">
                    <span className="dot" style={{ background: colorOf(pid) }} />
                    {pid}
                    {n && <span className="mono">击球 {n}</span>}
                  </span>
                );
              })
            ) : (
              <>
                {rate && (
                  <span>
                    回合胜率 <b className="mono">{rate}</b>
                  </span>
                )}
                {shots && (
                  <span>
                    击球 <b className="mono">{shots}</b>
                  </span>
                )}
                {!rate && !shots && <span>暂无个人数据</span>}
              </>
            )}
          </div>
        </div>
      </div>
    );
  };
  return (
    <div className="scoreboard">
      {side(sides[0], false)}
      <div className="sb-score">
        {board ? (
          <>
            <div className="big" aria-label={`比分 ${board.a} 比 ${board.b}`}>
              <span className={board.loser === 0 ? "lose" : ""}>{board.a}</span>
              <i>:</i>
              <span className={board.loser === 1 ? "lose" : ""}>{board.b}</span>
            </div>
            {board.games.length > 0 && (
              <div className="sb-games">
                {board.games.map((g) => (
                  <span key={g.game_no} className="chip mono" title={g.winner ? `${sideName(g.winner)} 胜` : "未完成"}>
                    第 {g.game_no} 局 {g.score[sides[0]] ?? 0}:{g.score[sides[1]] ?? 0}
                  </span>
                ))}
              </div>
            )}
            <div className="sb-line" title="按已标注的回合胜负推算（21 分制，领先 2 分获胜，30 分封顶）">
              {board.line}
            </div>
          </>
        ) : (
          <>
            <div className="big none" aria-label="暂无比分">
              <span>–</span>
              <i>:</i>
              <span>–</span>
            </div>
            <div className="sb-line hint">
              还没有标注回合胜负，在<a href={`#/videos/${encodeURIComponent(videoId)}`}>工作台</a>给回合设置得分方后显示比分
            </div>
          </>
        )}
      </div>
      {side(sides[1], true)}
    </div>
  );
}

function Warnings({ items }: { items: string[] | undefined }) {
  if (!items?.length) return null;
  return (
    <>
      {items.map((w) => (
        <div key={w} className="warn-line">
          <Icon name="alert" />
          <span>{w}</span>
        </div>
      ))}
    </>
  );
}

function CoachView({ coach, idx, onJump }: { coach: CoachReport; idx: Map<string, Evidence>; onJump: (k: string) => void }) {
  const [allAdvice, setAllAdvice] = useState(false);
  const advice = coach.shot_advice ?? [];
  const adviceFold = advice.length > ADVICE_ROWS + 2;
  return (
    <div className="coach-body">
      <p className="summary">{coach.summary}</p>
      <div className="insights">
        {coach.insights.map((i, n) => (
          <div key={n} className={`insight ${i.kind}`}>
            <div className="insight-head">
              <span className={`kind ${i.kind}`}>{KIND_LABEL[i.kind]}</span>
              {i.player_id && (
                <span className="who">
                  <span className="dot" style={{ background: colorOf(i.player_id) }} />
                  {i.player_id}
                </span>
              )}
              <b className="title">{i.title}</b>
              <span className="pills">
                <span className={`pill conf-${i.confidence}`}>{CONF_LABEL[i.confidence]}</span>
                {!i.grounded && <span className="pill needs_review">含未核实数字</span>}
              </span>
            </div>
            <div className="detail">{i.detail}</div>
            {i.evidence.length > 0 && (
              <div className="evidence">
                <span className="muted small">依据</span>
                {i.evidence.map((k) => {
                  const e = idx.get(k);
                  return (
                    <button key={k} className="chip" title={e ? evidenceText(e) : k} onClick={() => onJump(k)}>
                      {e ? e.label : k}
                    </button>
                  );
                })}
              </div>
            )}
            <Warnings items={i.warnings} />
          </div>
        ))}
      </div>
      {(coach.shot_advice?.length ?? 0) > 0 && (
        <>
          <div className="sub-head">
            <h3>逐拍建议</h3>
            <span className="sub">结合击球类型和双方站位找出的、可以换一种打法的击球。点时间可跳到那一拍回看。</span>
          </div>
          <ol className="training shot-advice">
            {(adviceFold && !allAdvice ? advice.slice(0, ADVICE_ROWS) : advice).map((a) => (
              <li key={a.moment_id}>
                <div className="advice-head">
                  <a className="tc" href={`#/videos/${coach.video_id}?t=${a.t_ms}`} title="跳到这一拍">
                    {a.timestamp}
                  </a>
                  <span className="small situation">{a.situation}</span>
                  <span className={`pill conf-${a.confidence}`}>{CONF_LABEL[a.confidence]}</span>
                </div>
                <div>
                  <b>{a.advice}</b>
                </div>
                <div className="muted small">{a.why}</div>
                <Warnings items={a.warnings} />
              </li>
            ))}
          </ol>
          {adviceFold && <MoreButton open={allAdvice} onToggle={() => setAllAdvice(!allAdvice)} more={`显示全部 ${advice.length} 条逐拍建议`} less={`只看前 ${ADVICE_ROWS} 条`} />}
        </>
      )}
      {coach.training.length > 0 && (
        <>
          <div className="sub-head">
            <h3>训练建议</h3>
          </div>
          <ol className="training">
            {coach.training.map((t, n) => (
              <li key={n}>
                {t.focus && <div className="small focus">针对：{t.focus}</div>}
                <div>
                  {t.player_id && (
                    <span className="who drill-who">
                      <span className="dot" style={{ background: colorOf(t.player_id) }} />
                      <b>{t.player_id}：</b>
                    </span>
                  )}
                  <b>{t.drill}</b> <span className="muted">— {t.why}</span>
                </div>
                {t.retest_metric && idx.get(t.retest_metric) && (
                  <div className="evidence">
                    <span className="small muted">下次复测</span>
                    <button className="chip" title={evidenceText(idx.get(t.retest_metric)!)} onClick={() => onJump(t.retest_metric!)}>
                      {idx.get(t.retest_metric)!.label}
                    </button>
                  </div>
                )}
                {(t.key_points?.length ?? 0) > 0 && (
                  <ul className="key-points small">
                    {t.key_points!.map((k) => (
                      <li key={k}>{k}</li>
                    ))}
                  </ul>
                )}
                {(t.sources?.length ?? 0) > 0 && (
                  <div className="row sources small">
                    <span className="muted">参考教学</span>
                    {t.sources!.map((r) => (
                      <a key={r.note_id} className="chip" href={videoLink(r)} target="_blank" rel="noreferrer" title={r.title}>
                        {r.coach.split(" (")[0]} · {r.title}
                        {r.drill ? ` · ${r.drill}` : ""}
                      </a>
                    ))}
                  </div>
                )}
                <Warnings items={t.warnings} />
              </li>
            ))}
          </ol>
        </>
      )}
    </div>
  );
}

/** Number and unit of a metric, split from exactly what fmtValue prints. */
function valueParts(m: Metric): { num: string; unit: string | null } {
  const s = fmtValue(m);
  if (m.value === null || !m.unit) return { num: s, unit: null };
  return m.unit === "%" ? { num: s.slice(0, -1), unit: "%" } : { num: s.slice(0, s.length - m.unit.length - 1), unit: m.unit };
}

/**
 * Column count that keeps the rows of a tile grid even: 7 tiles in a space for 6 become 4 + 3, not 6 + 1.
 * Before the first measurement the CSS auto-fill layout applies.
 */
function useBalancedColumns(count: number, min: number, gap = 10) {
  const [el, setEl] = useState<HTMLDivElement | null>(null);
  const [width, setWidth] = useState(0);
  useEffect(() => {
    if (!el) return;
    const measure = () => setWidth(el.clientWidth);
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, [el]);
  let style: CSSProperties | undefined;
  if (width > 0 && count > 0) {
    const fit = Math.max(1, Math.floor((width + gap) / (min + gap)));
    const rows = Math.ceil(count / fit);
    const cols = Math.min(fit, Math.max(Math.ceil(count / rows), Math.ceil(fit / 2)));
    style = { gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))` };
  }
  return [setEl, style] as const;
}

function MetricGrid({ metrics, focus, min = 168 }: { metrics: Metric[]; focus: string | null; min?: number }) {
  const [ref, style] = useBalancedColumns(metrics.length, min);
  return (
    <div className="metric-grid" ref={ref} style={style}>
      {metrics.map((m) => {
        const { num, unit } = valueParts(m);
        return (
          <div key={m.key} id={`m-${m.key}`} className={`metric ${m.status}${focus === m.key ? " focus" : ""}`} title={m.note ?? undefined}>
            <div className="label">{m.label}</div>
            <div className="value mono">
              {num}
              {unit && <small className={unit === "%" ? "pct" : undefined}>{unit}</small>}
            </div>
            <div className="foot">
              <span className={`status-dot ${m.status}`} />
              <span>{STATUS_LABEL[m.status]}</span>
              <span className="mono n">n={m.n}</span>
            </div>
            {m.note && <div className="note">{m.note}</div>}
          </div>
        );
      })}
    </div>
  );
}

function DistBars({ d, focus, color, bare }: { d: Distribution; focus: string | null; color?: string; bare?: boolean }) {
  const max = Math.max(1, ...Object.values(d.counts));
  return (
    <div id={`m-${d.key}`} className={`dist${focus === d.key ? " focus" : ""}`}>
      {/* bare: the card head above already carries the label, n and status */}
      {(!bare || d.note) && (
        <div className="dist-head">
          {!bare && (
            <>
              <b>{d.label}</b>
              <span className="muted mono">n={d.n}</span>
              <span className={`pill ${d.status}`}>{STATUS_LABEL[d.status]}</span>
            </>
          )}
          {d.note && <span className="muted">{d.note}</span>}
        </div>
      )}
      {d.n === 0 && <div className="muted small">暂无数据</div>}
      {d.n > 0 &&
        Object.entries(d.counts).map(([k, c]) => (
          <div key={k} className="dist-row">
            <span className="ellipsis" title={k}>
              {k}
            </span>
            <div className="bar">
              <i style={{ width: `${(100 * c) / max}%`, background: color }} />
            </div>
            <span className="mono">
              <b>{c}</b> ({Math.round((100 * c) / d.n)}%)
            </span>
          </div>
        ))}
    </div>
  );
}

/** A count split between exactly two players (doubles: "who plays the smash"). */
const isPairSplit = (d: Distribution) => {
  const keys = Object.keys(d.counts);
  return keys.length === 2 && keys.every((k) => /^P[1-4]$/.test(k));
};

/** One whole split between two partners: a single bar in the two players' colours, counts at both ends. */
function PairSplit({ d, focus }: { d: Distribution; focus: string | null }) {
  const [a, b] = Object.keys(d.counts);
  const ca = d.counts[a];
  const cb = d.counts[b];
  const pct = (c: number) => (d.n > 0 ? Math.round((100 * c) / d.n) : 0);
  return (
    <div id={`m-${d.key}`} className={`dist split${focus === d.key ? " focus" : ""}`}>
      <div className="split-label">
        <b>{d.label}</b>
        <span className="muted mono">n={d.n}</span>
      </div>
      <div className="split-row" role="img" aria-label={`${d.label}：${a} ${ca}，${b} ${cb}`}>
        <span className="who">
          <span className="dot" style={{ background: colorOf(a) }} />
          {a}
        </span>
        <span className="mono v">
          <b>{ca}</b> ({pct(ca)}%)
        </span>
        <span className="split-bar">
          {ca > 0 && <i style={{ flexGrow: ca, background: colorOf(a) }} />}
          {cb > 0 && <i style={{ flexGrow: cb, background: colorOf(b) }} />}
        </span>
        <span className="mono v">
          <b>{cb}</b> ({pct(cb)}%)
        </span>
        <span className="who">
          {b}
          <span className="dot" style={{ background: colorOf(b) }} />
        </span>
      </div>
      <span className={`pill ${d.status}`}>{STATUS_LABEL[d.status]}</span>
      {d.note && <div className="muted small split-note">{d.note}</div>}
    </div>
  );
}

function SubjectBody({ metrics, dists, color, focus }: { metrics: Metric[]; dists: Distribution[]; color: string; focus: string | null }) {
  return (
    <>
      {metrics.length > 0 && <MetricGrid metrics={metrics} focus={focus} />}
      {dists.length > 0 && (
        <div className="dist-list">
          {dists.map((d) => (isPairSplit(d) ? <PairSplit key={d.key} d={d} focus={focus} /> : <DistBars key={d.key} d={d} focus={focus} color={color} />))}
        </div>
      )}
    </>
  );
}

/** Cards for a set of players or teams, with one shared switch for which group of numbers they show. */
function SubjectSection({ id, title, hint, plans, tab, onTab, focus }: { id: string; title: string; hint: string; plans: SubjectPlan[]; tab: DataTab; onTab: (t: DataTab) => void; focus: string | null }) {
  const extra = plans.some((p) => p.extra);
  const groupIds = GROUP_ORDER.filter((g) => plans.some((p) => p.groups.some((x) => x.id === g)));
  const cur: DataTab = !extra ? "overview" : tab === "overview" || tab === "all" || groupIds.includes(tab) ? tab : "overview";
  const tabs: { id: DataTab; label: string }[] = [{ id: "overview", label: "概览" }, ...groupIds.map((g) => ({ id: g, label: GROUP_LABEL[g] })), { id: "all", label: "全部" }];
  return (
    <div id={id} className="subject-section">
      <div className="section-bar">
        <h2>{title}</h2>
        <span className="sub">{hint}</span>
        <span className="spacer" />
        {extra && (
          <div className="seg" role="group" aria-label={`${title}分组`}>
            {tabs.map((t) => (
              <button key={t.id} className={cur === t.id ? "on" : ""} aria-pressed={cur === t.id} onClick={() => onTab(t.id)}>
                {t.label}
              </button>
            ))}
          </div>
        )}
      </div>
      <div className="report-players">
        {plans.map((p) => {
          const color = colorOf(p.subject.id);
          const group = p.groups.find((g) => g.id === cur);
          return (
            <section className="card player-card" key={p.subject.id} style={{ "--pc": color } as CSSProperties}>
              <div className="player-head">
                <SideBadge id={p.subject.id} />
                <h2>{p.subject.name}</h2>
                {p.subject.sub && <span className="sub">{p.subject.sub}</span>}
                <span className="spacer" />
                {extra && (
                  <span className="sub mono">
                    {cur === "overview" ? `主要指标 ${p.headMetrics.length} / ${p.subject.metrics.length}` : cur === "all" ? `${p.subject.metrics.length} 项指标` : `${GROUP_LABEL[cur]} · ${group?.metrics.length ?? 0} 项`}
                  </span>
                )}
              </div>
              {cur === "overview" ? (
                <SubjectBody metrics={p.headMetrics} dists={p.headDists} color={color} focus={focus} />
              ) : cur === "all" ? (
                p.groups.map((g) => (
                  <Fragment key={g.id}>
                    <h3 className="group-head">{GROUP_LABEL[g.id]}</h3>
                    <SubjectBody metrics={g.metrics} dists={g.dists} color={color} focus={focus} />
                  </Fragment>
                ))
              ) : group ? (
                <SubjectBody metrics={group.metrics} dists={group.dists} color={color} focus={focus} />
              ) : (
                <p className="muted small">这一组没有数据。</p>
              )}
            </section>
          );
        })}
      </div>
    </div>
  );
}

function StyleCard({ st, videoId, idx, onJump }: { st: PlayerStyle; videoId: string; idx: Map<string, Evidence>; onJump: (k: string) => void }) {
  const colorId = st.kind === "singles" ? st.subject : st.players[0];
  const color = colorOf(colorId);
  const ranked = st.scores.filter((x) => x.score !== null);
  const phases = st.phases.filter((p) => p.shots > 0);
  const tactics = st.tactics.filter((t) => t.uses > 0);
  return (
    <div className="style-card" style={{ "--pc": color } as CSSProperties}>
      <div className="style-head">
        <SideBadge id={colorId} label={st.subject} />
        <div className="style-title">
          <span className="style-who">{styleWho(st)}</span>
          <span className="style-label">{st.label}</span>
        </div>
        <span className="spacer" />
        <span className={`pill ${st.status}`}>{STATUS_LABEL[st.status]}</span>
      </div>
      <div className="muted small">
        依据 {st.n_shots} 拍已分类击球{st.note ? `；${st.note}` : ""}
      </div>
      {st.status !== "undeterminable" && (
        <>
          {st.primary && st.primary.score !== null && (
            <div className="style-primary">
              最接近 <b>{st.primary.name}</b>（{st.primary.name_en}，代表：{st.primary.representatives.map((r) => r.split(" ")[0]).join("、")}）
              {st.primary.reasons.length > 0 && <span className="muted"> · 依据：{st.primary.reasons.join("、")}</span>}
            </div>
          )}
          <div className="style-scores">
            {ranked.map((x) => (
              <div key={x.code} className="dist-row" title={x.reasons.join("、")}>
                <span className="ellipsis">{x.name}</span>
                <div className="bar">
                  <i style={{ width: `${x.score}%`, background: color }} />
                </div>
                <span className="mono">{Math.round(x.score!)}</span>
              </div>
            ))}
          </div>
          <h3>攻防状态</h3>
          <div className="phase-list">
            {phases.map((p) => (
              <div key={p.phase} className="phase-row">
                <span className="kind">{PHASE_LABEL[p.phase]}</span>
                <span className="mono shots">{p.shots} 拍</span>
                <span className="mix">
                  {Object.entries(p.counts)
                    .map(([k, v]) => `${shotZh(k)} ${v}`)
                    .join("，")}
                </span>
                {p.recommended_pct !== null && <span className="muted rec">符合推荐打法 {Math.round(p.recommended_pct)}%</span>}
              </div>
            ))}
          </div>
          {tactics.length > 0 && (
            <>
              <h3>战术套路</h3>
              <div className="table-scroll">
                <table className="style-tactics small">
                  <thead>
                    <tr>
                      <th>套路</th>
                      <th className="num">次数</th>
                      <th className="num">胜率</th>
                      <th className="num">逼对手被动</th>
                      <th>回看</th>
                    </tr>
                  </thead>
                  <tbody>
                    {tactics.map((t) => (
                      <tr key={t.code}>
                        <td>{t.name}</td>
                        <td className="num mono">{t.uses}</td>
                        <td className="num mono">{t.success_pct === null ? "—" : `${Math.round(t.success_pct)}%（${t.won}/${t.decided}）`}</td>
                        <td className="num mono">{t.forced_pct === null ? "—" : `${Math.round(t.forced_pct)}%`}</td>
                        <td>
                          <div className="tc-list">
                            {t.examples_ms.map((ms) => (
                              <a key={ms} className="tc" href={`#/videos/${encodeURIComponent(videoId)}?t=${ms}`}>
                                {clock(ms)}
                              </a>
                            ))}
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          )}
          {st.habits.length > 0 && (
            <>
              <h3>回球习惯（预测下一拍）</h3>
              <ul className="small style-list">
                {st.habits.map((h) => (
                  <li key={h.after}>
                    接{shotZh(h.after)}时 {Math.round(h.pct)}% 回{shotZh(h.reply)}（{h.n}/{h.of}）
                  </li>
                ))}
              </ul>
            </>
          )}
          {st.advice.length > 0 && (
            <>
              <h3>建议</h3>
              <ul className="small style-list">
                {st.advice.map((a) => (
                  <li key={a}>{a}</li>
                ))}
              </ul>
            </>
          )}
          {st.evidence.length > 0 && (
            <div className="evidence style-evidence">
              <span className="muted small">特征</span>
              {st.evidence.map((k) => (
                <button key={k} className="chip" title={idx.get(k) ? evidenceText(idx.get(k)!) : k} onClick={() => onJump(k)}>
                  {idx.get(k)?.label ?? k}
                </button>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}
