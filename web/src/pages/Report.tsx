import { useCallback, useEffect, useMemo, useState } from "react";
import { PlayerCompare, RallyTable, ScoreFlow } from "../components/MatchFlow";
import { MemoryCommit } from "../components/MemoryCommit";
import { api } from "../lib/api";
import { download } from "../lib/export";
import { colorOf } from "../lib/players";
import { clock, CONF_LABEL, evidenceText, fmtValue, indexStats, KIND_LABEL, PHASE_LABEL, reportMarkdown, shotZh, STATUS_LABEL, styleWho, videoLink, type Evidence } from "../lib/report";
import type { CoachReport, Distribution, MatchReport, Metric, PlayerStyle, Video } from "../lib/types";

export function Report({ videoId }: { videoId: string }) {
  const [video, setVideo] = useState<Video | null>(null);
  const [report, setReport] = useState<MatchReport | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [generating, setGenerating] = useState(false);
  const [coachError, setCoachError] = useState<string | null>(null);
  const [focus, setFocus] = useState<string | null>(null);

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

  const jump = (key: string) => {
    setFocus(key);
    document.getElementById(`m-${key}`)?.scrollIntoView({ behavior: "smooth", block: "center" });
  };

  if (error) {
    return (
      <main className="page">
        <div className="card">
          无法生成报告：{error}
          <div className="muted small" style={{ marginTop: 6 }}>
            需要先完成分析（回合与击球事件）。<a href={`#/videos/${encodeURIComponent(videoId)}`}>回到工作台</a>
          </div>
        </div>
      </main>
    );
  }
  if (!report) {
    return (
      <main className="page">
        <div className="card">加载报告…</div>
      </main>
    );
  }
  const { stats, coach } = report;

  return (
    <main className="page report">
      <div className="row">
        <a href={`#/videos/${encodeURIComponent(videoId)}`}>← 工作台</a>
        <h1 className="report-title">赛后报告</h1>
        <span className="muted" title={video?.filename}>
          {video?.filename}
        </span>
        <span className="spacer" />
        <button onClick={() => doExport("md")}>导出 Markdown</button>
        <button onClick={() => doExport("json")}>导出 JSON</button>
        {api.reportExportUrl(videoId, "html") && (
          <button onClick={() => doExport("html")} title="一个独立的网页文件：图表都在文件里，离线可看，可直接发给教练或球友">
            导出网页报告
          </button>
        )}
      </div>

      <section className="card">
        <div className="row" style={{ marginBottom: 8 }}>
          <h2 style={{ margin: 0 }}>AI 教练点评</h2>
          {coach && (
            <span className="muted small">
              {coach.provider}
              {coach.model ? ` · ${coach.model}` : ""} · {new Date(coach.created_at).toLocaleString()}
            </span>
          )}
          <span className="spacer" />
          <button className="primary" disabled={generating} onClick={() => void generate()}>
            {generating ? "生成中…（本地模型可能需要几分钟）" : coach ? "重新生成" : "生成 AI 点评"}
          </button>
        </div>
        {coachError && <div className="error">生成失败：{coachError}</div>}
        {report.coach_stale && <div className="notice warn">这份点评生成后数据又有修正（例如改了击球者或胜负），内容可能已过时，建议重新生成。</div>}
        {coach?.fallback_reason && (
          <div className="notice">
            {coach.provider === "offline" ? "AI 模型不可用，改用离线规则模板" : `已改用 ${coach.provider}`}：{coach.fallback_reason}
          </div>
        )}
        {coach ? (
          <CoachView coach={coach} idx={idx} onJump={jump} />
        ) : (
          <p className="muted">
            点评会严格依据下面的统计数据生成，每条观点都标明引用的指标和可信度；数据不足时会直接说“证据不足”。
          </p>
        )}
      </section>

      {(stats.styles?.length ?? 0) > 0 && (
        <section className="card">
          <h2>打法风格</h2>
          <div className="muted small">
            按击球类型、攻防状态、节奏和双方站位，用规则和打法原型做匹配（没有用标注数据训练）。吻合度 0–100，参考区间来自职业单打统计，业余比赛看倾向即可。
          </div>
          <div className="report-players">
            {stats.styles!.map((st) => (
              <StyleCard key={st.subject} st={st} videoId={videoId} idx={idx} onJump={jump} />
            ))}
          </div>
        </section>
      )}

      {(coach?.caveats ?? stats.caveats).length > 0 && (
        <section className="card">
          <h2>数据说明</h2>
          <ul className="caveats">
            {(coach?.caveats ?? stats.caveats).map((c) => (
              <li key={c}>{c}</li>
            ))}
          </ul>
        </section>
      )}

      <section className="card">
        <h2>全场统计</h2>
        <MetricGrid metrics={stats.metrics} focus={focus} />
        {stats.distributions.map((d) => (
          <DistBars key={d.key} d={d} focus={focus} />
        ))}
      </section>

      <ScoreFlow stats={stats} />
      <PlayerCompare stats={stats} />
      <RallyTable stats={stats} videoId={videoId} />

      <div className="report-players">
        {stats.players.map((p) => (
          <section className="card" key={p.player_id} style={{ borderTop: `3px solid ${colorOf(p.player_id)}` }}>
            <h2>{p.player_id}</h2>
            <MetricGrid metrics={p.metrics} focus={focus} />
            {p.distributions.map((d) => (
              <DistBars key={d.key} d={d} focus={focus} color={colorOf(p.player_id)} />
            ))}
          </section>
        ))}
      </div>
      <div className="muted small">
        状态按样本量和球场标定情况给出：可用 / 待复核 / 不可判定。统计只使用修正后的事件，原始模型预测不会被改写。指纹 {stats.fingerprint}
      </div>
      <MemoryCommit videoId={videoId} playerIds={stats.players.map((p) => p.player_id)} />
    </main>
  );
}

function CoachView({ coach, idx, onJump }: { coach: CoachReport; idx: Map<string, Evidence>; onJump: (k: string) => void }) {
  return (
    <div className="coach">
      <p className="summary">{coach.summary}</p>
      <div className="insights">
        {coach.insights.map((i, n) => (
          <div key={n} className={`insight ${i.kind}`}>
            <div className="row">
              <span className={`kind ${i.kind}`}>{KIND_LABEL[i.kind]}</span>
              {i.player_id && (
                <span className="row" style={{ gap: 4 }}>
                  <span className="dot" style={{ background: colorOf(i.player_id) }} />
                  {i.player_id}
                </span>
              )}
              <b>{i.title}</b>
              <span className="spacer" />
              <span className={`pill conf-${i.confidence}`}>{CONF_LABEL[i.confidence]}</span>
              {!i.grounded && <span className="pill needs_review">含未核实数字</span>}
            </div>
            <div className="detail">{i.detail}</div>
            {i.evidence.length > 0 && (
              <div className="row evidence">
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
            {i.warnings.map((w) => (
              <div key={w} className="small" style={{ color: "var(--warn)" }}>
                ⚠ {w}
              </div>
            ))}
          </div>
        ))}
      </div>
      {(coach.shot_advice?.length ?? 0) > 0 && (
        <>
          <h3>逐拍建议</h3>
          <div className="muted small">结合击球类型、双方站位和脚步找出的、可以换一种打法或步法的击球。点时间可跳到那一拍回看。</div>
          <ol className="training shot-advice">
            {coach.shot_advice!.map((a) => (
              <li key={a.moment_id}>
                <div className="row">
                  <a className="chip mono" href={`#/videos/${coach.video_id}?t=${a.t_ms}`} title="跳到这一拍">
                    {a.timestamp}
                  </a>
                  <span className="small">{a.situation}</span>
                  <span className="spacer" />
                  <span className={`pill conf-${a.confidence}`}>{CONF_LABEL[a.confidence]}</span>
                </div>
                <div>
                  <b>{a.advice}</b>
                </div>
                <div className="muted small">{a.why}</div>
                {a.footwork && <div className="small">👣 脚步：{a.footwork}</div>}
                {a.warnings.map((w) => (
                  <div key={w} className="small" style={{ color: "var(--warn)" }}>
                    ⚠ {w}
                  </div>
                ))}
              </li>
            ))}
          </ol>
        </>
      )}
      {coach.training.length > 0 && (
        <>
          <h3>训练建议</h3>
          <ol className="training">
            {coach.training.map((t, n) => (
              <li key={n}>
                {t.focus && <div className="small focus">针对：{t.focus}</div>}
                {t.player_id && <b>{t.player_id}：</b>}
                <b>{t.drill}</b> <span className="muted">— {t.why}</span>
                {t.retest_metric && idx.get(t.retest_metric) && (
                  <span className="small muted">
                    （下次复测：
                    <button className="chip" onClick={() => onJump(t.retest_metric!)}>
                      {idx.get(t.retest_metric)!.label}
                    </button>
                    ）
                  </span>
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
                {t.warnings?.map((w) => (
                  <div key={w} className="small" style={{ color: "var(--warn)" }}>
                    ⚠ {w}
                  </div>
                ))}
              </li>
            ))}
          </ol>
        </>
      )}
    </div>
  );
}

function MetricGrid({ metrics, focus }: { metrics: Metric[]; focus: string | null }) {
  return (
    <div className="metric-grid">
      {metrics.map((m) => (
        <div key={m.key} id={`m-${m.key}`} className={`metric ${m.status}${focus === m.key ? " focus" : ""}`} title={m.note ?? undefined}>
          <div className="muted small">{m.label}</div>
          <div className="mono value">{fmtValue(m)}</div>
          <div className="small muted">
            n={m.n} · <span className={`pill ${m.status}`}>{STATUS_LABEL[m.status]}</span>
          </div>
          {m.note && <div className="small muted note">{m.note}</div>}
        </div>
      ))}
    </div>
  );
}

function DistBars({ d, focus, color }: { d: Distribution; focus: string | null; color?: string }) {
  const max = Math.max(1, ...Object.values(d.counts));
  return (
    <div id={`m-${d.key}`} className={`dist${focus === d.key ? " focus" : ""}`}>
      <div className="row small">
        <span>{d.label}</span>
        <span className="muted">n={d.n}</span>
        <span className={`pill ${d.status}`}>{STATUS_LABEL[d.status]}</span>
        {d.note && <span className="muted">{d.note}</span>}
      </div>
      {d.n > 0 &&
        Object.entries(d.counts).map(([k, c]) => (
          <div key={k} className="dist-row small">
            <span>{k}</span>
            <div className="bar">
              <i style={{ width: `${(100 * c) / max}%`, background: color }} />
            </div>
            <span className="mono muted">
              {c}（{Math.round((100 * c) / d.n)}%）
            </span>
          </div>
        ))}
    </div>
  );
}

function StyleCard({ st, videoId, idx, onJump }: { st: PlayerStyle; videoId: string; idx: Map<string, Evidence>; onJump: (k: string) => void }) {
  const color = colorOf(st.kind === "singles" ? st.subject : st.players[0]);
  const ranked = st.scores.filter((x) => x.score !== null);
  return (
    <div className="style-card" style={{ borderTop: `3px solid ${color}` }}>
      <div className="row">
        <b>{styleWho(st)}</b>
        <span className="style-label">{st.label}</span>
        <span className="spacer" />
        <span className={`pill ${st.status}`}>{STATUS_LABEL[st.status]}</span>
      </div>
      <div className="muted small">
        依据 {st.n_shots} 拍已分类击球{st.note ? `；${st.note}` : ""}
      </div>
      {st.status !== "undeterminable" && (
        <>
          {st.primary && st.primary.score !== null && (
            <div className="small">
              最接近 <b>{st.primary.name}</b>（{st.primary.name_en}，代表：{st.primary.representatives.map((r) => r.split(" ")[0]).join("、")}）
              {st.primary.reasons.length > 0 && <span className="muted"> · 依据：{st.primary.reasons.join("、")}</span>}
            </div>
          )}
          <div className="style-scores">
            {ranked.map((x) => (
              <div key={x.code} className="dist-row small" title={x.reasons.join("、")}>
                <span>{x.name}</span>
                <div className="bar">
                  <i style={{ width: `${x.score}%`, background: color }} />
                </div>
                <span className="mono muted">{Math.round(x.score!)}</span>
              </div>
            ))}
          </div>
          <h3>攻防状态</h3>
          {st.phases
            .filter((p) => p.shots > 0)
            .map((p) => (
              <div key={p.phase} className="small">
                <b>{PHASE_LABEL[p.phase]}</b> {p.shots} 拍：
                {Object.entries(p.counts)
                  .map(([k, v]) => `${shotZh(k)} ${v}`)
                  .join("，")}
                {p.recommended_pct !== null && <span className="muted">（符合推荐打法 {Math.round(p.recommended_pct)}%）</span>}
              </div>
            ))}
          {st.tactics.some((t) => t.uses > 0) && (
            <>
              <h3>战术套路</h3>
              <table className="style-tactics small">
                <thead>
                  <tr>
                    <th>套路</th>
                    <th>次数</th>
                    <th>胜率</th>
                    <th>逼对手被动</th>
                    <th>回看</th>
                  </tr>
                </thead>
                <tbody>
                  {st.tactics
                    .filter((t) => t.uses > 0)
                    .map((t) => (
                      <tr key={t.code}>
                        <td>{t.name}</td>
                        <td className="mono">{t.uses}</td>
                        <td className="mono">{t.success_pct === null ? "—" : `${Math.round(t.success_pct)}%（${t.won}/${t.decided}）`}</td>
                        <td className="mono">{t.forced_pct === null ? "—" : `${Math.round(t.forced_pct)}%`}</td>
                        <td>
                          {t.examples_ms.map((ms) => (
                            <a key={ms} className="chip mono" href={`#/videos/${encodeURIComponent(videoId)}?t=${ms}`}>
                              {clock(ms)}
                            </a>
                          ))}
                        </td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </>
          )}
          {st.habits.length > 0 && (
            <>
              <h3>回球习惯（预测下一拍）</h3>
              <ul className="small">
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
              <ul className="small">
                {st.advice.map((a) => (
                  <li key={a}>{a}</li>
                ))}
              </ul>
            </>
          )}
          {st.evidence.length > 0 && (
            <div className="row evidence">
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
