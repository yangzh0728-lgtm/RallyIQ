import type { Point } from "./types";

// Standard badminton court, metres. x across (0..6.1), y along (0..13.4), y=0 = baseline nearer the camera
// (same convention as rallyiq/contracts/geometry.py).
export const COURT_W = 6.1;
export const COURT_L = 13.4;
const SIDE_ALLEY = 0.46;
const SHORT_SERVICE = 1.98; // from net
const LONG_SERVICE_DOUBLES = 0.76; // from baseline
const NET_Y = COURT_L / 2;

/** Court line segments in meters. */
export function courtLinesM(): [Point, Point][] {
  const L: [Point, Point][] = [];
  const h = (y: number, x0 = 0, x1 = COURT_W) => L.push([[x0, y], [x1, y]]);
  const v = (x: number, y0 = 0, y1 = COURT_L) => L.push([[x, y0], [x, y1]]);
  h(0);
  h(COURT_L);
  h(LONG_SERVICE_DOUBLES);
  h(COURT_L - LONG_SERVICE_DOUBLES);
  h(NET_Y - SHORT_SERVICE);
  h(NET_Y + SHORT_SERVICE);
  v(0);
  v(COURT_W);
  v(SIDE_ALLEY);
  v(COURT_W - SIDE_ALLEY);
  v(COURT_W / 2, 0, NET_Y - SHORT_SERVICE);
  v(COURT_W / 2, NET_Y + SHORT_SERVICE, COURT_L);
  return L;
}

export const NET_LINE_M: [Point, Point] = [[0, NET_Y], [COURT_W, NET_Y]];

/** Outer corners, in the order PUT /calibration/manual expects: near-left, near-right, far-right, far-left. */
export const CORNERS_M: Point[] = [
  [0, 0],
  [COURT_W, 0],
  [COURT_W, COURT_L],
  [0, COURT_L],
];
export const CORNER_NAMES = ["近左", "近右", "远右", "远左"];

/**
 * Solve the 3x3 homography H (row-major, h33 = 1) mapping src[i] -> dst[i] for 4 point pairs.
 * Returns null for degenerate input.
 */
export function homographyFrom4(src: Point[], dst: Point[]): number[] | null {
  if (src.length !== 4 || dst.length !== 4) return null;
  const A: number[][] = [];
  for (let i = 0; i < 4; i++) {
    const [x, y] = src[i];
    const [u, v] = dst[i];
    A.push([x, y, 1, 0, 0, 0, -u * x, -u * y, u]);
    A.push([0, 0, 0, x, y, 1, -v * x, -v * y, v]);
  }
  const h = solve8(A);
  return h ? [...h, 1] : null;
}

/** Gaussian elimination with partial pivoting on an 8x9 augmented matrix. */
function solve8(M: number[][]): number[] | null {
  const n = 8;
  const a = M.map((r) => r.slice());
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(a[r][c]) > Math.abs(a[p][c])) p = r;
    if (Math.abs(a[p][c]) < 1e-12) return null;
    [a[c], a[p]] = [a[p], a[c]];
    for (let r = 0; r < n; r++) {
      if (r === c) continue;
      const f = a[r][c] / a[c][c];
      for (let k = c; k <= n; k++) a[r][k] -= f * a[c][k];
    }
  }
  return a.map((r, i) => r[n] / r[i]);
}

export function applyH(H: number[], [x, y]: Point): Point {
  const w = H[6] * x + H[7] * y + H[8];
  return [(H[0] * x + H[1] * y + H[2]) / w, (H[3] * x + H[4] * y + H[5]) / w];
}

export function invertH(H: number[]): number[] | null {
  const [a, b, c, d, e, f, g, h, i] = H;
  const A = e * i - f * h;
  const B = -(d * i - f * g);
  const C = d * h - e * g;
  const det = a * A + b * B + c * C;
  if (Math.abs(det) < 1e-12) return null;
  const inv = [
    A, -(b * i - c * h), b * f - c * e,
    B, a * i - c * g, -(a * f - c * d),
    C, -(a * h - b * g), a * e - b * d,
  ].map((v) => v / det);
  return inv.map((v) => v / inv[8]);
}

/** In-play rally boundary: singles uses the inner sidelines, doubles the outer ones (back boundary = baseline). */
export function boundaryLinesM(format: "singles" | "doubles"): [Point, Point][] {
  const x0 = format === "doubles" ? 0 : SIDE_ALLEY;
  const c: Point[] = [
    [x0, 0],
    [COURT_W - x0, 0],
    [COURT_W - x0, COURT_L],
    [x0, COURT_L],
  ];
  return c.map((p, i) => [p, c[(i + 1) % 4]] as [Point, Point]);
}

/** Segments in court metres projected to image px through H (court meters -> image). */
export function projectLines(H: number[], segs: [Point, Point][]): [Point, Point][] {
  return segs.map(([p, q]) => [applyH(H, p), applyH(H, q)].map(([x, y]) => [r1(x), r1(y)]) as [Point, Point]);
}

/** Court lines projected to image px through H (court meters -> image). */
export function projectCourtLines(H: number[]): [Point, Point][] {
  return projectLines(H, [...courtLinesM(), NET_LINE_M]);
}
const r1 = (v: number) => Math.round(v * 10) / 10;
