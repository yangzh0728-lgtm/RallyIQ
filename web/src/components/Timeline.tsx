import { useState } from "react";
import type { PlaybackClock } from "../lib/clock";
import { formatMs } from "../lib/corrections";
import { useClockMs, useClockPlaying } from "../lib/useClock";
import { colorOf } from "../lib/players";
import { FOOTWORK_MARK, KEY_KINDS } from "../lib/footwork";
import { SHOT_TYPE_COLOR, SHOT_TYPE_ZH, typeLabel } from "../lib/strokes";
import type { FootworkEvent, Rally, ShotEvent } from "../lib/types";

interface Props {
  durationMs: number;
  rallies: Rally[];
  shots: ShotEvent[];
  clock: PlaybackClock;
  selectedRallyId: string | null;
  selectedShotId: string | null;
  onSelectRally: (id: string) => void;
  onSelectShot: (id: string) => void;
  rallyNo: (id: string) => number;
  fps: number;
  /** Footwork actions (split step, lunge, jump) shown as a row under the shots. */
  footwork?: FootworkEvent[];
}

const RATES = [0.25, 0.5, 1, 2];

export function Timeline(p: Props) {
  const t = useClockMs(p.clock);
  const playing = useClockPlaying(p.clock);
  const [zoomRally, setZoomRally] = useState(false);
  const sel = p.rallies.find((r) => r.rally_id === p.selectedRallyId);
  const [v0, v1] = zoomRally && sel ? [Math.max(0, sel.start_ms - 1000), Math.min(p.durationMs, sel.end_ms + 1000)] : [0, p.durationMs];
  const pct = (ms: number) => `${((ms - v0) / (v1 - v0)) * 100}%`;
  const inView = (ms: number) => ms >= v0 && ms <= v1;
  const frameMs = 1000 / p.fps;

  return (
    <div style={{ display: "grid", gap: 6 }}>
      <div className="controls">
        <button onClick={() => p.clock.toggle()} style={{ width: 64 }}>
          {playing ? "暂停" : "播放"}
        </button>
        <button title="上一帧 (←)" onClick={() => p.clock.seek(t - frameMs)}>
          ◀︎ 帧
        </button>
        <button title="下一帧 (→)" onClick={() => p.clock.seek(t + frameMs)}>
          帧 ▶︎
        </button>
        <span className="mono" style={{ minWidth: 130 }}>
          {formatMs(t)} / {formatMs(p.durationMs)}
        </span>
        <label>
          速度
          <select value={p.clock.rate} onChange={(e) => p.clock.setRate(Number(e.target.value))}>
            {RATES.map((r) => (
              <option key={r} value={r}>
                {r}×
              </option>
            ))}
          </select>
        </label>
        <button className={zoomRally ? "active" : ""} disabled={!sel} onClick={() => setZoomRally((z) => !z)}>
          {zoomRally ? "显示整场" : "放大当前回合"}
        </button>
        <span className="spacer" />
        <span className="muted small">
          <kbd>空格</kbd> 播放 <kbd>←</kbd>
          <kbd>→</kbd> 逐帧 <kbd>J</kbd>
          <kbd>K</kbd> 上/下一拍 <kbd>[</kbd>
          <kbd>]</kbd> 上/下回合
        </span>
      </div>
      <div
        className="timeline"
        onClick={(e) => {
          const r = e.currentTarget.getBoundingClientRect();
          p.clock.seek(v0 + ((e.clientX - r.left) / r.width) * (v1 - v0));
        }}
      >
        {p.rallies
          .filter((r) => r.end_ms >= v0 && r.start_ms <= v1)
          .map((r) => (
            <div
              key={r.rally_id}
              className={`tl-rally ${r.status} ${r.source} ${r.rally_id === p.selectedRallyId ? "selected" : ""}`}
              style={{ left: pct(Math.max(v0, r.start_ms)), width: `calc(${pct(Math.min(v1, r.end_ms))} - ${pct(Math.max(v0, r.start_ms))})` }}
              title={`回合 ${p.rallyNo(r.rally_id)} · ${formatMs(r.start_ms)}–${formatMs(r.end_ms)}${r.status === "needs_review" ? " · 待复核" : ""}`}
              onClick={(e) => {
                e.stopPropagation();
                p.onSelectRally(r.rally_id);
              }}
            >
              {p.rallyNo(r.rally_id)}
            </div>
          ))}
        {p.shots
          .filter((s) => inView(s.t_ms))
          .map((s) => (
            <div
              key={s.shot_id}
              className={`tl-shot ${s.status} ${s.shot_id === p.selectedShotId ? "selected" : ""}`}
              style={{ left: pct(s.t_ms), background: colorOf(s.hitter) }}
              title={`${formatMs(s.t_ms)} · ${s.hitter ?? "击球者未知"} · 置信度 ${Math.round(s.confidence * 100)}%${
                s.stroke ? ` · ${typeLabel(s.shot_type)}${s.stroke.source === "model" && s.stroke.status !== "ok" ? "（类型待复核）" : ""}` : ""
              }`}
              onClick={(e) => {
                e.stopPropagation();
                p.onSelectShot(s.shot_id);
              }}
            />
          ))}
        {zoomRally &&
          p.shots
            .filter((s) => inView(s.t_ms) && s.shot_type)
            .map((s) => (
              <span key={`t-${s.shot_id}`} className="tl-type" style={{ left: pct(s.t_ms), color: SHOT_TYPE_COLOR[s.shot_type!] }}>
                {SHOT_TYPE_ZH[s.shot_type!]}
              </span>
            ))}
        {(p.footwork ?? [])
          .filter((e) => KEY_KINDS.includes(e.kind) && inView(e.t_ms))
          .map((e) => (
            <span
              key={e.event_id}
              className={`tl-foot ${e.kind}`}
              style={{ left: pct(e.t_ms), color: colorOf(e.player_id), opacity: 0.45 + 0.55 * e.confidence }}
              title={`${formatMs(e.t_ms)} · ${e.player_id} · ${e.label}`}
              onClick={(ev) => {
                ev.stopPropagation();
                p.clock.pause();
                p.clock.seek(Math.max(0, e.t_ms - 400));
              }}
            >
              {FOOTWORK_MARK[e.kind]}
            </span>
          ))}
        {inView(t) && <div className="tl-head" style={{ left: pct(t) }} />}
      </div>
    </div>
  );
}
