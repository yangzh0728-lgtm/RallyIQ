import { useMemo } from "react";
import type { PlaybackClock } from "../lib/clock";
import { formatMs } from "../lib/corrections";
import { FOOTWORK_ZH, LANDING_ZH, SPLIT_ZH, summarize } from "../lib/footwork";
import { colorOf } from "../lib/players";
import type { FootworkResult, PlayerId, Rally } from "../lib/types";

interface Props {
  footwork: FootworkResult | null;
  error: string | null;
  players: PlayerId[];
  rally: Rally | undefined;
  rallyNo: number | null;
  selectedShotId: string | null;
  clock: PlaybackClock;
}

/** 脚步：每位球员的启动步 / 回位概况，选中那一拍的脚步，当前回合的步法事件。 */
export function FootworkPanel({ footwork, error, players, rally, rallyNo, selectedShotId, clock }: Props) {
  const shot = footwork?.shots.find((s) => s.shot_id === selectedShotId) ?? null;
  const rallyEvents = useMemo(
    () => (footwork && rally ? footwork.events.filter((e) => e.t_ms >= rally.start_ms - 300 && e.t_ms <= rally.end_ms + 300) : []),
    [footwork, rally],
  );
  if (error) return <div className="panel small"><h3>脚步</h3><div className="muted">脚步数据不可用：{error}</div></div>;
  if (!footwork) return <div className="panel small"><h3>脚步</h3><div className="muted">加载中…</div></div>;
  const quality = new Map(footwork.quality.map((q) => [q.player_id, q]));
  return (
    <div className="panel small">
      <h3>脚步</h3>
      <div className="muted" style={{ marginBottom: 6 }}>
        从脚踝（有脚部模型时加上脚跟和脚尖）投影到球场：实心脚印 = 正在着地，空心 = 刚离地，脚印上的黑点是脚尖。启动步 = 对手击球前后双脚同时轻跳落地。
      </div>
      {players.map((pid) => {
        const s = summarize(footwork, pid);
        const q = quality.get(pid);
        return (
          <div key={pid} style={{ marginBottom: 6 }}>
            <div className="row" style={{ gap: 6 }}>
              <span className="dot" style={{ background: colorOf(pid) }} />
              <b>{pid}</b>
              {q && q.status !== "ok" && (
                <span className="pill needs_review" title={q.jitter_m != null ? `脚部定位误差约 ${q.jitter_m.toFixed(2)} 米` : undefined}>
                  只作参考
                </span>
              )}
            </div>
            <div className="fw-grid">
              <span className="muted">启动步</span>
              <span className="mono">{s.splitPct == null ? "—" : `${s.splitPct}%（及时 ${s.onTimePct}%）`}</span>
              <span className="muted">起动</span>
              <span className="mono">{s.reactionMs == null ? "—" : `${Math.round(s.reactionMs)} ms`}</span>
              <span className="muted">及时回中心</span>
              <span className="mono">{s.recoveredPct == null ? "—" : `${s.recoveredPct}%`}</span>
              <span className="muted">弓步 / 起跳</span>
              <span className="mono">
                {s.lunges} / {s.jumps}
              </span>
              {s.landingsJudged > 0 && (
                <>
                  <span className="muted" title="移动中每一步落地时脚的哪一部分先着地（由脚跟和脚尖关键点判断）">着地方式</span>
                  <span className="mono">
                    前脚掌 {s.forefootPct}% · 脚跟 {s.heelFirstPct}%
                  </span>
                </>
              )}
            </div>
          </div>
        );
      })}
      {shot && (
        <div style={{ borderTop: "1px solid var(--line)", paddingTop: 6, marginTop: 4 }}>
          <div className="row" style={{ gap: 6 }}>
            <b>这一拍</b>
            <span className="muted">
              {shot.player_id} · 启动步{SPLIT_ZH[shot.split_step]}
              {shot.lunge_landing ? ` · 弓步${LANDING_ZH[shot.lunge_landing]}` : ""}
            </span>
          </div>
          <div>{shot.summary || "这一拍没有可用的脚步数据"}</div>
        </div>
      )}
      {rally && (
        <>
          <div className="muted" style={{ marginTop: 6 }}>
            第 {rallyNo} 回合的步法（点击跳转）
          </div>
          <div className="fw-events">
            {rallyEvents.length === 0 && <div className="muted">没有识别出步法事件</div>}
            {rallyEvents.map((e) => (
              <div
                key={e.event_id}
                className="fw-event"
                title={e.label}
                onClick={() => {
                  clock.pause();
                  clock.seek(Math.max(0, e.t_ms - 400));
                }}
              >
                <span className="mono muted">{formatMs(e.t_ms).replace(/\.\d+$/, "")}</span>
                <span style={{ color: colorOf(e.player_id) }}>{e.player_id}</span>
                <span>
                  <b>{FOOTWORK_ZH[e.kind]}</b> <span className="muted">{e.label}</span>
                </span>
              </div>
            ))}
          </div>
        </>
      )}
      {footwork.caveats.length > 0 && (
        <div className="muted" style={{ marginTop: 6 }} title={footwork.caveats.join("\n")}>
          ⚠ {footwork.caveats[0]}
          {footwork.caveats.length > 1 ? `（另有 ${footwork.caveats.length - 1} 条说明）` : ""}
        </div>
      )}
    </div>
  );
}
