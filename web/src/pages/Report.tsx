import { useCallback, useEffect, useMemo, useState } from "react";
import { MemoryCommit } from "../components/MemoryCommit";
import { api } from "../lib/api";
import { download } from "../lib/export";
import { colorOf } from "../lib/players";
import { CONF_LABEL, evidenceText, fmtValue, indexStats, KIND_LABEL, reportMarkdown, STATUS_LABEL, videoLink, type Evidence } from "../lib/report";
import type { CoachReport, Distribution, MatchReport, Metric, Video } from "../lib/types";

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

  const doExport = (format: "md" | "json") => {
    const url = api.reportExportUrl(videoId, format);
    if (url) {
      window.location.href = url;
      return;
    }
    if (!report) return;
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
