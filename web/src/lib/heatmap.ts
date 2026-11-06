import { COURT_L, COURT_W } from "./court";
import type { OverlayFrame, PlayerId } from "./types";

export interface HeatGrid {
  cols: number;
  rows: number;
  cell: number;
  counts: number[][];
  samples: number;
  max: number;
  distanceM: number;
}

/**
 * Same grid as GET /api/videos/{id}/heatmap (counts[iy][ix], cell (ix,iy) covers x in [ix*cell,(ix+1)*cell)),
 * computed client-side from the overlay so switching player/rally is instant and also yields distance.
 */
/** Max plausible movement between consecutive frames; larger jumps are ID swaps / glitches. */
const MAX_STEP_M = 0.6;

/** ``who`` is one player or several (a doubles team, or everyone); distance is summed per player. */
export function computeHeat(frames: OverlayFrame[], who: PlayerId | PlayerId[], fromMs: number, toMs: number, cell = 0.5): HeatGrid {
  const ids = new Set(Array.isArray(who) ? who : [who]);
  const cols = Math.ceil(COURT_W / cell);
  const rows = Math.ceil(COURT_L / cell);
  const counts = Array.from({ length: rows }, () => new Array<number>(cols).fill(0));
  let samples = 0;
  let max = 0;
  let distanceM = 0;
  const prev = new Map<PlayerId, [number, number]>();
  for (const f of frames) {
    if (f.t_ms < fromMs || f.t_ms > toMs) continue;
    for (const id of ids) {
      const xy = f.players.find((p) => p.player_id === id)?.foot_court;
      if (!xy) {
        prev.delete(id);
        continue;
      }
      const [x, y] = xy;
      const last = prev.get(id);
      if (last) {
        const d = Math.hypot(x - last[0], y - last[1]);
        if (d < MAX_STEP_M) distanceM += d;
      }
      prev.set(id, [x, y]);
      // Allow a small margin outside the lines (players stand behind the baseline).
      const c = Math.floor(x / cell);
      const r = Math.floor(y / cell);
      if (c < 0 || r < 0 || c >= cols || r >= rows) continue;
      counts[r][c] += 1;
      samples += 1;
      if (counts[r][c] > max) max = counts[r][c];
    }
  }
  return { cols, rows, cell, counts, samples, max, distanceM };
}
