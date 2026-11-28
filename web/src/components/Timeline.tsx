import { useState, useSyncExternalStore, type CSSProperties } from "react";
import type { PlaybackClock } from "../lib/clock";
import { formatMs } from "../lib/corrections";
import { useClockMs, useClockPlaying } from "../lib/useClock";
import { colorOf } from "../lib/players";
import { SHOT_TYPE_COLOR, SHOT_TYPE_ZH, typeLabel } from "../lib/strokes";
import type { Rally, ShotEvent } from "../lib/types";
import { Icon } from "./ui";

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
}

const RATES = [0.25, 0.5, 1, 2];

/** Ruler steps in seconds, each with how many unlabelled subdivisions it gets. */
const RULER_STEPS: [number, number][] = [
  [0.1, 1], [0.2, 2], [0.5, 5], [1, 5], [2, 4], [5, 5], [10, 5], [15, 3], [30, 3], [60, 4], [120, 4], [300, 5], [600, 5], [900, 3], [1800, 3], [3600, 4],
];
const MAX_LABELS = 12;

function tickLabel(ms: number, fine: boolean): string {
  const m = Math.floor(ms / 60000);
  const s = (ms - m * 60000) / 1000;
  return fine ? `${m}:${s.toFixed(1).padStart(4, "0")}` : `${m}:${String(Math.round(s)).padStart(2, "0")}`;
}

/** Labelled ticks on a "nice" step (at most about MAX_LABELS across the visible range) plus minor ticks between them. */
function rulerTicks(v0: number, v1: number): { major: { ms: number; label: string }[]; minor: number[] } {
  const span = v1 - v0;
  if (!(span > 0)) return { major: [], minor: [] };
  const [stepS, parts] = RULER_STEPS.find(([s]) => span / (s * 1000) <= MAX_LABELS) ?? RULER_STEPS[RULER_STEPS.length - 1];
  const step = Math.round(stepS * 1000);
  const sub = step / parts;
  const major: { ms: number; label: string }[] = [];
  const minor: number[] = [];
  for (let k = Math.ceil(v0 / sub - 1e-6); k * sub <= v1 + 1e-6; k++) {
    const ms = Math.round(k * sub);
    if (k % parts === 0) major.push({ ms, label: tickLabel(ms, stepS < 1) });
    else minor.push(ms);
  }
  return { major, minor };
}

export function Timeline(p: Props) {
  const t = useClockMs(p.clock);
  const playing = useClockPlaying(p.clock);
  // The clock emits on setRate, so the speed switch follows it (including rate changes made elsewhere).
  const rate = useSyncExternalStore(
    (cb) => p.clock.subscribe(cb),
    () => p.clock.rate,
  );
  const [zoomRally, setZoomRally] = useState(false);
  const sel = p.rallies.find((r) => r.rally_id === p.selectedRallyId);
  const zoomed = zoomRally && !!sel;
  const [v0, v1] = zoomRally && sel ? [Math.max(0, sel.start_ms - 1000), Math.min(p.durationMs, sel.end_ms + 1000)] : [0, p.durationMs];
  const pct = (ms: number) => `${((ms - v0) / (v1 - v0)) * 100}%`;
  const inView = (ms: number) => ms >= v0 && ms <= v1;
  const frameMs = 1000 / p.fps;
  const ticks = rulerTicks(v0, v1);
  // A label needs room to its right; the last few percent of the ruler only get the tick mark.
  const roomForLabel = (ms: number) => (ms - v0) / (v1 - v0) < 0.975;

  return (
    <div className="tl-wrap">
      <div className="transport">
        <button className="play" aria-label={playing ? "暂停" : "播放"} title={playing ? "暂停（空格）" : "播放（空格）"} onClick={() => p.clock.toggle()}>
          <Icon name={playing ? "pause" : "play"} />
        </button>
        <div className="step-btns">
          <button className="icon-btn" title="上一帧 (←)" aria-label="上一帧" onClick={() => p.clock.seek(t - frameMs)}>
            <Icon name="stepBack" />
          </button>
          <button className="icon-btn" title="下一帧 (→)" aria-label="下一帧" onClick={() => p.clock.seek(t + frameMs)}>
            <Icon name="stepForward" />
          </button>
        </div>
        <span className="timecode" aria-label="当前时间 / 总时长">
          {formatMs(t)} <span>/ {formatMs(p.durationMs)}</span>
        </span>
        <div className="seg" role="group" aria-label="播放速度" title="播放速度">
          {RATES.map((r) => (
            <button key={r} className={rate === r ? "on" : ""} aria-pressed={rate === r} onClick={() => p.clock.setRate(r)}>
              {r}×
            </button>
          ))}
        </div>
        <button className={zoomRally ? "active" : ""} disabled={!sel} aria-pressed={zoomRally} onClick={() => setZoomRally((z) => !z)}>
          <Icon name="zoom" />
          {zoomRally ? "显示整场" : "放大当前回合"}
        </button>
        <span className="spacer" />
        <div className="keys">
          <span>
            <kbd>空格</kbd>播放
          </span>
          <span>
            <kbd>←</kbd>
            <kbd>→</kbd>逐帧
          </span>
          <span>
            <kbd>J</kbd>
            <kbd>K</kbd>上/下一拍
          </span>
          <span>
            <kbd>[</kbd>
            <kbd>]</kbd>上/下回合
          </span>
        </div>
      </div>
      <div className={`tl${zoomed ? " zoomed" : ""}`}>
        <span />
        <div className="tl-ruler" aria-hidden="true">
          {ticks.minor.map((ms) => (
            <i key={ms} style={{ left: pct(ms) }} />
          ))}
          {ticks.major.map((m) => (
            <span key={m.ms} style={{ left: pct(m.ms) }}>
              {roomForLabel(m.ms) ? m.label : ""}
            </span>
          ))}
        </div>
        <div className="tl-lanes" aria-hidden="true">
          <span>回合</span>
          <span>击球</span>
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
                style={
                  {
                    left: pct(Math.max(v0, r.start_ms)),
                    width: `calc(${pct(Math.min(v1, r.end_ms))} - ${pct(Math.max(v0, r.start_ms))})`,
                    ...(r.winner ? { "--wc": colorOf(r.winner) } : null),
                  } as CSSProperties
                }
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
          {inView(t) && <div className="tl-head" style={{ left: pct(t) }} />}
        </div>
      </div>
    </div>
  );
}
