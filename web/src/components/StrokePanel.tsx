import { useState } from "react";
import type { PlaybackClock } from "../lib/clock";
import { formatMs } from "../lib/corrections";
import { colorOf, labelOf } from "../lib/players";
import { SHOT_TYPE_COLOR, SHOT_TYPE_ZH, detailChips, strokeStats } from "../lib/strokes";
import { useClockMs } from "../lib/useClock";
import { PLAYER_IDS, type Rally, type ShotEvent } from "../lib/types";
import { TypeChip } from "./CorrectionPanel";

interface Props {
  shots: ShotEvent[]; // effective shots of the whole video
  rally: Rally | undefined;
  rallyNo: number | null;
  clock: PlaybackClock;
}

/** Latest stroke at the playhead and each player's stroke-type distribution (whole video or current rally). */
export function StrokePanel({ shots, rally, rallyNo, clock }: Props) {
  const t = useClockMs(clock);
  const [scope, setScope] = useState<"all" | "rally">("all");
  const inScope = scope === "rally" && rally ? shots.filter((s) => s.rally_id === rally.rally_id) : shots;
  const stats = strokeStats(inScope, PLAYER_IDS);
  const latest = rally ? [...shots].reverse().find((s) => s.rally_id === rally.rally_id && s.t_ms <= t + 1) : undefined;
  const anyTyped = shots.some((s) => s.stroke);

  return (
    <div className="panel small">
      <div className="row" style={{ marginBottom: 6 }}>
        <h3 style={{ margin: 0 }}>击球类型</h3>
        <span className="spacer" />
        <button className={scope === "all" ? "active" : ""} onClick={() => setScope("all")}>
          整场
        </button>
        <button className={scope === "rally" ? "active" : ""} disabled={!rally} onClick={() => setScope("rally")}>
          {rally ? `回合 ${rallyNo}` : "当前回合"}
        </button>
      </div>
      {!anyTyped && <div className="muted">还没有击球类型结果（重新分析后会出现）。</div>}
      {anyTyped && (
        <>
          <div className="row" style={{ marginBottom: 8 }}>
            <span className="muted">最近一拍</span>
            <span className="spacer" />
            {latest ? (
              <>
                <span className="dot" style={{ background: colorOf(latest.hitter) }} />
                <span>{labelOf(latest.hitter)}</span>
                <TypeChip shot={latest} />
                <span className="muted mono">{formatMs(latest.t_ms)}</span>
              </>
            ) : (
              <span className="muted">—</span>
            )}
          </div>
          {latest?.stroke?.detail && (
            <div className="stroke-detail">
              <div className="chips">
                {detailChips(latest.stroke).map((c) => (
                  <span key={c.key} className={`chip${c.inferred ? " inferred" : ""}`} title={c.title}>
                    {c.label}
                  </span>
                ))}
              </div>
              {latest.stroke.detail.description_zh && <div>{latest.stroke.detail.description_zh}</div>}
              <div className="muted">实线标签 = 模型识别或测量；虚线标签 = 按规则推断，仅供参考。鼠标悬停看来源。</div>
            </div>
          )}
          {stats.map((st) => {
            const max = Math.max(1, ...st.counts.map((c) => c.n));
            return (
              <div key={st.player} style={{ marginBottom: 8 }}>
                <div className="row">
                  <span className="dot" style={{ background: colorOf(st.player) }} />
                  <b>{labelOf(st.player)}</b>
                  <span className="spacer" />
                  <span className="muted">
                    {st.typed}/{st.total} 拍有类型{st.review ? ` · ${st.review} 待复核` : ""}
                  </span>
                </div>
                <div className="stroke-bars">
                  {st.counts.length === 0 && <span className="muted">—</span>}
                  {st.counts.map((c) => (
                    <div key={c.type} className="stroke-bar" title={`${SHOT_TYPE_ZH[c.type]}：${c.n} 拍，其中 ${c.review} 拍待复核（浅色）`}>
                      <span>{SHOT_TYPE_ZH[c.type]}</span>
                      <span className="track" style={{ width: `${(c.n / max) * 100}%` }}>
                        <i style={{ width: `${((c.n - c.review) / c.n) * 100}%`, background: SHOT_TYPE_COLOR[c.type] }} />
                        <i className="review" style={{ width: `${(c.review / c.n) * 100}%`, background: SHOT_TYPE_COLOR[c.type] }} />
                      </span>
                      <span className="mono muted">
                        {c.n} · {Math.round((c.n / Math.max(1, st.typed)) * 100)}%
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            );
          })}
          <div className="muted">深色 = 可用或人工确认，浅色 = 待复核。样本少时比例不可靠。</div>
        </>
      )}
    </div>
  );
}
