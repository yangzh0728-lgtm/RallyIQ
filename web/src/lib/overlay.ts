import { CORNER_NAMES } from "./court";
import type { CourtLines, OverlayData, OverlayFrame, PlayerId, Point } from "./types";

/** Index of the last frame with t_ms <= t (binary search; frames are sorted by time). */
export function frameIndexAt(frames: { t_ms: number }[], t: number): number {
  let lo = 0;
  let hi = frames.length - 1;
  if (hi < 0) return -1;
  if (t <= frames[0].t_ms) return 0;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (frames[mid].t_ms <= t) lo = mid;
    else hi = mid - 1;
  }
  return lo;
}

export function courtAt(court: CourtLines[], frame: number): CourtLines | undefined {
  return court.find((c) => frame >= c.start_frame && frame <= c.end_frame);
}

export function pointInPolygon(poly: Point[], [x, y]: Point): boolean {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i];
    const [xj, yj] = poly[j];
    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

/** The other court (4 corners) under an image point, for "analyse this court instead". */
export function otherCourtAt(cal: CourtLines | undefined, p: Point): Point[] | null {
  return cal?.other_courts?.find((c) => pointInPolygon(c, p)) ?? null;
}

/** Courts to keep ignoring after switching to ``chosen``: the rest plus the old target. */
export function courtsAfterSwitch(cal: CourtLines, chosen: Point[]): Point[][] {
  const rest = (cal.other_courts ?? []).filter((c) => c !== chosen);
  return cal.image_corners ? [...rest, cal.image_corners] : rest;
}

// COCO-17 limb pairs (fallback when the overlay has no `skeleton`).
export const COCO_LIMBS: [number, number][] = [
  [5, 7], [7, 9], [6, 8], [8, 10], [5, 6], [5, 11], [6, 12], [11, 12],
  [11, 13], [13, 15], [12, 14], [14, 16], [0, 5], [0, 6],
];

export interface Layers {
  ai: boolean; // master switch: original vs AI-annotated view
  skeleton: boolean;
  bbox: boolean;
  shuttle: boolean;
  court: boolean;
  /** Pure abstract view: hide the original frame, draw only the overlays on a dark background. */
  abstract: boolean;
  opacity: number; // 0..1
}

export const DEFAULT_LAYERS: Layers = { ai: true, skeleton: true, bbox: false, shuttle: true, court: true, abstract: false, opacity: 0.9 };

const KP_MIN_CONF = 0.3;
const TRAIL_FRAMES = 14;

export function drawOverlay(
  ctx: CanvasRenderingContext2D,
  data: OverlayData,
  t: number,
  layers: Layers,
  colorOf: (playerId: PlayerId | null) => string,
  calibPoints: Point[] | null,
  choosingCourt = false,
) {
  const { canvas } = ctx;
  const sx = canvas.width / data.video.width;
  const sy = canvas.height / data.video.height;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.setTransform(sx, 0, 0, sy, 0, 0);
  const px = 1 / Math.min(sx, sy); // one screen pixel in source px

  if (calibPoints) {
    drawCalibPoints(ctx, calibPoints, px);
    return;
  }
  if (!layers.ai) return;
  ctx.globalAlpha = layers.opacity;
  const idx = frameIndexAt(data.frames, t);
  const f: OverlayFrame | undefined = data.frames[idx];

  if ((layers.court || choosingCourt) && f) {
    const cal = courtAt(data.court, f.frame);
    // Other courts in view are ignored by the analysis: outline them faintly so it is
    // visible which court was analysed, and highlight them when the user is choosing.
    for (const oc of cal?.other_courts ?? []) {
      ctx.strokeStyle = choosingCourt ? "#22d3ee" : "#94a3b880";
      ctx.fillStyle = "#22d3ee22";
      ctx.lineWidth = (choosingCourt ? 2.5 : 1) * px;
      ctx.setLineDash([8 * px, 6 * px]);
      ctx.beginPath();
      oc.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
      ctx.closePath();
      ctx.stroke();
      if (choosingCourt) ctx.fill();
      ctx.setLineDash([]);
    }
    if (cal) {
      const shaky = cal.status !== "ok";
      ctx.strokeStyle = shaky ? "#f59e0b" : "#ffffffb0";
      ctx.lineWidth = 1.5 * px;
      ctx.setLineDash(shaky ? [6 * px, 4 * px] : []);
      ctx.beginPath();
      for (const [[x1, y1], [x2, y2]] of cal.lines) {
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
      }
      ctx.stroke();
      // The in-play boundary for this format (singles: inner sidelines, doubles: outer) drawn bolder.
      if (cal.boundary?.length) {
        ctx.lineWidth = 3 * px;
        ctx.beginPath();
        for (const [[x1, y1], [x2, y2]] of cal.boundary) {
          ctx.moveTo(x1, y1);
          ctx.lineTo(x2, y2);
        }
        ctx.stroke();
      }
      ctx.setLineDash([]);
    }
  }

  if (f && (layers.skeleton || layers.bbox)) {
    const limbs = data.skeleton?.length ? data.skeleton : COCO_LIMBS;
    for (const p of f.players) {
      const color = colorOf(p.player_id);
      // Short-gap fills are not detections: draw them fainter.
      ctx.globalAlpha = layers.opacity * (p.interpolated ? 0.45 : 1);
      if (layers.bbox && p.bbox) {
        const [x1, y1, x2, y2] = p.bbox;
        ctx.strokeStyle = color;
        ctx.lineWidth = 1 * px;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      }
      if (layers.skeleton && p.keypoints?.length) {
        const k = p.keypoints;
        ctx.strokeStyle = color;
        ctx.lineWidth = 2.5 * px;
        ctx.lineCap = "round";
        ctx.beginPath();
        for (const [a, b] of limbs) {
          if (!k[a] || !k[b] || k[a][2] < KP_MIN_CONF || k[b][2] < KP_MIN_CONF) continue;
          ctx.moveTo(k[a][0], k[a][1]);
          ctx.lineTo(k[b][0], k[b][1]);
        }
        ctx.stroke();
        ctx.fillStyle = color;
        for (const [x, y, c] of k) {
          if (c < KP_MIN_CONF) continue;
          ctx.beginPath();
          ctx.arc(x, y, 2.5 * px, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    }
  }

  ctx.globalAlpha = layers.opacity;
  if (layers.shuttle && idx >= 0) {
    // Trail: detected = solid, interpolated = translucent + dashed (an estimate, not an observation).
    const from = Math.max(0, idx - TRAIL_FRAMES);
    for (let i = from + 1; i <= idx; i++) {
      const a = data.frames[i - 1].shuttle;
      const b = data.frames[i].shuttle;
      if (!a || !b) continue;
      const age = (idx - i) / TRAIL_FRAMES;
      ctx.globalAlpha = layers.opacity * (1 - age * 0.8) * (b.source === "interpolated" ? 0.45 : 1);
      ctx.strokeStyle = "#facc15";
      ctx.lineWidth = 2.5 * px;
      ctx.setLineDash(b.source === "interpolated" ? [4 * px, 4 * px] : []);
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.stroke();
    }
    ctx.setLineDash([]);
    const s = f?.shuttle;
    if (s) {
      ctx.globalAlpha = layers.opacity * (s.source === "interpolated" ? 0.5 : 1);
      ctx.fillStyle = "#facc15";
      ctx.strokeStyle = "#000";
      ctx.lineWidth = 1 * px;
      ctx.beginPath();
      ctx.arc(s.x, s.y, 4.5 * px, 0, Math.PI * 2);
      if (s.source === "interpolated") {
        ctx.setLineDash([2 * px, 2 * px]);
        ctx.strokeStyle = "#facc15";
        ctx.stroke();
        ctx.setLineDash([]);
      } else {
        ctx.fill();
        ctx.stroke();
      }
    }
  }
  ctx.globalAlpha = 1;
}

function drawCalibPoints(ctx: CanvasRenderingContext2D, pts: Point[], px: number) {
  ctx.strokeStyle = "#38bdf8";
  ctx.fillStyle = "#38bdf8";
  ctx.lineWidth = 2 * px;
  ctx.beginPath();
  pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
  if (pts.length === 4) ctx.closePath();
  ctx.stroke();
  ctx.font = `${12 * px}px sans-serif`;
  pts.forEach(([x, y], i) => {
    ctx.beginPath();
    ctx.arc(x, y, 5 * px, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillText(CORNER_NAMES[i], x + 8 * px, y - 8 * px);
  });
}

export const CALIB_ORDER_HINT = `依次点击球场外框四角：${CORNER_NAMES.join(" → ")}`;
