import type { PlaybackClock } from "../lib/clock";
import { frameIndexAt } from "../lib/overlay";
import { colorOf } from "../lib/players";
import { useClockMs } from "../lib/useClock";
import type { OverlayData } from "../lib/types";
import { CourtView } from "./CourtView";

const TRAIL_MS = 2000;

export function MiniMap({ overlay, clock }: { overlay: OverlayData; clock: PlaybackClock }) {
  const t = useClockMs(clock);
  const idx = frameIndexAt(overlay.frames, t);
  const from = frameIndexAt(overlay.frames, t - TRAIL_MS);
  const frame = overlay.frames[idx];
  const trails = new Map<string, string>();
  for (let i = from; i <= idx && i >= 0; i += 2) {
    for (const p of overlay.frames[i].players) {
      if (!p.foot_court || !p.player_id) continue;
      const prev = trails.get(p.player_id);
      trails.set(p.player_id, `${prev ?? ""}${prev ? "L" : "M"}${p.foot_court[0]},${p.foot_court[1]} `);
    }
  }
  return (
    <div className="panel">
      <h3>球场小地图</h3>
      <CourtView height={220}>
        {[...trails].map(([id, d]) => (
          <path key={id} d={d} fill="none" stroke={colorOf(id)} strokeWidth={0.06} opacity={0.6} />
        ))}
        {frame?.players.map((p, i) =>
          p.foot_court ? (
            <circle key={p.player_id ?? i} cx={p.foot_court[0]} cy={p.foot_court[1]} r={0.22} fill={colorOf(p.player_id)} stroke="#000" strokeWidth={0.04} opacity={p.interpolated ? 0.5 : 1} />
          ) : null,
        )}
      </CourtView>
      {frame && frame.players.every((p) => !p.foot_court) && <div className="muted small">此刻没有可用的足点（未标定或未检测到）</div>}
    </div>
  );
}
