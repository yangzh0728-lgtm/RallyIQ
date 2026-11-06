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

export function computeHeat(frames: OverlayFrame[], playerId: PlayerId, fromMs: number, toMs: number, cell = 0.5): HeatGrid {
  const cols = Math.ceil(COURT_W / cell);
  const rows = Math.ceil(COURT_L / cell);
  const counts = Array.from({ length: rows }, () => new Array<number>(cols).fill(0));
  let samples = 0;
  let max = 0;
  let distanceM = 0;
  let prev: [number, number] | null = null;
  for (const f of frames) {
    if (f.t_ms < fromMs || f.t_ms > toMs) continue;
    const xy = f.players.find((p) => p.player_id === playerId)?.foot_court;
    if (!xy) {
      prev = null;
      continue;
    }
    const [x, y] = xy;
    if (prev) {
      const d = Math.hypot(x - prev[0], y - prev[1]);
      if (d < MAX_STEP_M) distanceM += d;
    }
    prev = [x, y];
    // Allow a small margin outside the lines (players stand behind the baseline).
    const c = Math.floor(x / cell);
    const r = Math.floor(y / cell);
    if (c < 0 || r < 0 || c >= cols || r >= rows) continue;
    counts[r][c] += 1;
    samples += 1;
    if (counts[r][c] > max) max = counts[r][c];
  }
  return { cols, rows, cell, counts, samples, max, distanceM };
}
