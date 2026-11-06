// Deterministic synthetic analysis in the backend's wire format, so the UI can be demoed without
// the Python service (open the app with ?mock).
import { applyH, boundaryLinesM, COURT_L, COURT_W, CORNERS_M, homographyFrom4, projectCourtLines, projectLines } from "./court";
import { COCO_LIMBS } from "./overlay";
import { TEAM_OF, type EventsResult, type Keypoint, type OverlayData, type OverlayFrame, type PlayerId, type Point, type Rally, type ShotEvent, type VideoMeta } from "./types";

function rng(seed: number) {
  let s = seed >>> 0 || 1;
  return () => {
    s ^= s << 13;
    s ^= s >>> 17;
    s ^= s << 5;
    return (s >>> 0) / 4294967296;
  };
}

interface Hit {
  t: number;
  hitter: number; // index into PID; even = near team (P1, P3), odd = far team (P2, P4)
  pos: Point; // court metres
}

export interface MockAnalysis {
  overlay: OverlayData;
  events: EventsResult;
}

const PID: PlayerId[] = ["P1", "P2", "P3", "P4"];

export function generateMock(
  videoId: string,
  opts: { width?: number; height?: number; fps?: number; durationMs?: number; filename?: string; doubles?: boolean } = {},
): MockAnalysis {
  const doubles = !!opts.doubles;
  const nPlayers = doubles ? 4 : 2;
  const width = opts.width ?? 1280;
  const height = opts.height ?? 720;
  const fps = opts.fps ?? 30;
  const duration = Math.max(10_000, Math.round(opts.durationMs ?? 75_000));
  const frameOf = (t: number) => Math.round((t * fps) / 1000);
  const tOf = (f: number) => Math.round((f * 1000) / fps);
  const r = rng(hash(videoId));

  // Back-centre phone view: court metres -> image px. Corner order: near-left, near-right, far-right, far-left.
  const sx = width / 1280;
  const sy = height / 720;
  const imgCorners: Point[] = [
    [210 * sx, 660 * sy],
    [1070 * sx, 660 * sy],
    [810 * sx, 175 * sy],
    [470 * sx, 175 * sy],
  ];
  const H = homographyFrom4(CORNERS_M, imgCorners)!;

  const rallies: Rally[] = [];
  const hits: Hit[][] = [];
  let t = 1500 + r() * 1500;
  while (t < duration - 4000) {
    const n = 3 + Math.floor(r() * 9);
    const rh: Hit[] = [];
    let team = r() < 0.5 ? 0 : 1;
    let ht = t + 300;
    for (let i = 0; i < n && ht < duration - 1500; i++) {
      // Doubles: the partner whose lane (left/right half of the court) the shuttle comes to hits it.
      const lane = r() < 0.5 ? 0 : 1;
      const hitter = doubles ? team + 2 * lane : team;
      const x = doubles ? (lane === (team === 0 ? 0 : 1) ? 0.6 + r() * 2.3 : 3.2 + r() * 2.3) : 0.8 + r() * (COURT_W - 1.6);
      rh.push({ t: ht, hitter, pos: [x, team === 0 ? 0.5 + r() * 5.3 : 7.6 + r() * 5.3] });
      ht += 850 + r() * 650;
      team = 1 - team;
    }
    if (rh.length < 2) break;
    const end = rh[rh.length - 1].t + 900 + r() * 500;
    const conf = 0.5 + r() * 0.5;
    const sf = frameOf(t);
    const ef = frameOf(end);
    rallies.push({
      rally_id: `r${String(rallies.length + 1).padStart(3, "0")}`,
      start_frame: sf,
      end_frame: ef,
      start_ms: tOf(sf),
      end_ms: tOf(ef),
      shot_ids: [],
      end_reason: r() < 0.7 ? "shuttle_lost" : "shuttle_static",
      winner: r() < 0.2 ? null : doubles ? (r() < 0.5 ? "A" : "B") : PID[r() < 0.5 ? 0 : 1],
      confidence: round2(conf),
      status: conf < 0.65 ? "needs_review" : "ok",
      source: "model",
    });
    hits.push(rh);
    t = end + 3500 + r() * 4000;
  }

  // Shots, with a few realistic model errors to correct.
  const shots: ShotEvent[] = [];
  const addShot = (ri: number, tm: number, hitter: PlayerId, conf: number, reasons: string[], forceReview = false) => {
    const f = frameOf(tm);
    const id = `s${String(shots.length + 1).padStart(4, "0")}`;
    shots.push({
      shot_id: id, rally_id: rallies[ri].rally_id, frame: f, t_ms: tOf(f), hitter, hitter_confidence: round2(0.5 + r() * 0.5),
      shuttle_px: null, shuttle_court: null, hitter_court: null, shot_type: null, target_zone: null,
      confidence: round2(conf), status: conf < 0.6 || forceReview ? "needs_review" : "ok", source: "model", reasons,
    });
    rallies[ri].shot_ids.push(id);
  };
  hits.forEach((rh, ri) => {
    rh.forEach((h, i) => {
      if (ri % 3 === 1 && i === 2) return; // missed hit (false negative)
      const wrongHitter = ri % 4 === 2 && i === 1;
      addShot(ri, h.t + (r() - 0.5) * 60, PID[wrongHitter ? h.hitter ^ 1 : h.hitter], 0.45 + r() * 0.55, ["mock: trajectory turn", "mock: wrist near shuttle"], wrongHitter);
    });
    if (ri % 5 === 3) {
      const last = rh[rh.length - 1];
      addShot(ri, last.t + 500, PID[last.hitter ^ 1], 0.41, ["mock: trajectory turn"]); // false positive
    }
  });
  shots.sort((a, b) => a.frame - b.frame);

  // Per-frame overlay.
  const home: Point[] = doubles
    ? [
        [1.6, 3.2], // P1 near-left
        [4.5, 10.2], // P2 far (image right)
        [4.5, 3.2], // P3 near-right
        [1.6, 10.2], // P4 far (image left)
      ]
    : [
        [COURT_W / 2, 3.2],
        [COURT_W / 2, 10.2],
      ];
  const slots = Array.from({ length: nPlayers }, (_, i) => i);
  const frames: OverlayFrame[] = [];
  const nFrames = Math.floor((duration / 1000) * fps);
  let gapUntil = -1;
  for (let f = 0; f < nFrames; f++) {
    const tm = (f * 1000) / fps;
    const ri = rallies.findIndex((ra) => tm >= ra.start_ms - 800 && tm <= ra.end_ms + 600);
    const rh = ri >= 0 ? hits[ri] : null;
    const pos = slots.map((p) => playerPos(p, tm, rh, home[p], f));
    let shuttle: OverlayFrame["shuttle"] = null;
    if (rh && tm >= rh[0].t - 200 && tm <= rallies[ri].end_ms) {
      const sp = shuttlePos(tm, rh, H);
      if (sp) {
        if (gapUntil < f && r() < 0.012) gapUntil = f + 4 + Math.floor(r() * 8);
        const interp = f <= gapUntil;
        shuttle = { x: round1(sp[0]), y: round1(sp[1]), source: interp ? "interpolated" : "detected", confidence: interp ? 0.3 : round2(0.75 + r() * 0.25) };
      }
    }
    const swing = (p: number) => {
      const h = rh?.find((x) => x.hitter === p && Math.abs(x.t - tm) < 250);
      return h ? 1 - Math.abs(h.t - tm) / 250 : 0;
    };
    frames.push({
      frame: f,
      t_ms: tOf(f),
      shuttle,
      players: slots.map((p) => {
        const foot = applyH(H, pos[p]);
        const s = pxPerMeter(H, pos[p]);
        return {
          player_id: PID[p],
          team: TEAM_OF[PID[p]],
          side: p % 2 === 0 ? "near" : "far",
          bbox: [round1(foot[0] - 0.45 * s), round1(foot[1] - 1.85 * s), round1(foot[0] + 0.45 * s), round1(foot[1])],
          keypoints: skeleton(foot, s, swing(p), p % 2 === 0, f),
          foot_court: [round2(pos[p][0]), round2(pos[p][1])],
          interpolated: false,
        };
      }),
    });
  }

  const video: VideoMeta = {
    video_id: videoId,
    filename: opts.filename ?? "demo.mp4",
    fps,
    frame_count: nFrames,
    width,
    height,
    duration_ms: duration,
    codec: "h264",
    rotation: 0,
    variable_frame_rate: false,
  };
  return {
    overlay: {
      schema_version: "0.1.0",
      video,
      match_format: doubles ? "doubles" : "singles",
      court: [
        {
          start_frame: 0,
          end_frame: nFrames - 1,
          lines: projectCourtLines(H),
          boundary: projectLines(H, boundaryLinesM(doubles ? "doubles" : "singles")),
          status: "ok",
        },
      ],
      keypoint_format: "coco17",
      skeleton: COCO_LIMBS,
      frames,
    },
    events: { schema_version: "0.1.0", video_id: videoId, match_format: doubles ? "doubles" : "singles", rallies, shots },
  };
}

function playerPos(p: number, t: number, rh: Hit[] | null, home: Point, f: number): Point {
  const jitter: Point = [Math.sin(f / 23 + p) * 0.08, Math.cos(f / 31 + p) * 0.08];
  if (!rh) return [home[0] + jitter[0], home[1] + jitter[1]];
  // Keyframes: own hits at hit position, otherwise drift back toward home.
  const keys: { t: number; pos: Point }[] = [{ t: rh[0].t - 800, pos: home }];
  for (const h of rh) {
    if (h.hitter === p) keys.push({ t: h.t, pos: h.pos });
    else keys.push({ t: h.t, pos: [home[0] * 0.6 + keys[keys.length - 1].pos[0] * 0.4, home[1]] });
  }
  keys.push({ t: rh[rh.length - 1].t + 1500, pos: home });
  let i = keys.findIndex((k) => k.t > t);
  if (i <= 0) i = i === 0 ? 1 : keys.length - 1;
  const a = keys[i - 1];
  const b = keys[i];
  const u = Math.min(1, Math.max(0, (t - a.t) / Math.max(1, b.t - a.t)));
  const e = u * u * (3 - 2 * u);
  return [a.pos[0] + (b.pos[0] - a.pos[0]) * e + jitter[0], a.pos[1] + (b.pos[1] - a.pos[1]) * e + jitter[1]];
}

function shuttlePos(t: number, rh: Hit[], H: number[]): Point | null {
  let i = rh.findIndex((h) => h.t > t);
  let a: Hit;
  let b: { t: number; pos: Point };
  if (i === -1) {
    a = rh[rh.length - 1];
    // Last shot lands deep on the other side.
    b = { t: a.t + 1000, pos: [a.pos[0] * 0.5 + 1.5, a.hitter % 2 === 0 ? COURT_L - 1.2 : 1.2] };
    i = rh.length;
  } else if (i === 0) return null;
  else {
    a = rh[i - 1];
    b = rh[i];
  }
  const u = Math.min(1, (t - a.t) / Math.max(1, b.t - a.t));
  const g: Point = [a.pos[0] + (b.pos[0] - a.pos[0]) * u, a.pos[1] + (b.pos[1] - a.pos[1]) * u];
  const endH = i === rh.length ? 0 : 2.2;
  const hgt = 2.2 + (endH - 2.2) * u + 4.5 * u * (1 - u);
  const [x, y] = applyH(H, g);
  return [x, y - hgt * pxPerMeter(H, g)];
}

function pxPerMeter(H: number[], p: Point): number {
  const a = applyH(H, p);
  const b = applyH(H, [p[0] + 1, p[1]]);
  return Math.hypot(b[0] - a[0], b[1] - a[1]);
}

/** COCO-17 pose standing on `foot`, `s` px per metre, with a simple racket-arm swing. */
function skeleton(foot: Point, s: number, swing: number, near: boolean, f: number): Keypoint[] {
  const [fx, fy] = foot;
  const bob = Math.sin(f / 4) * 0.02 * s;
  // A far player faces the camera, so left/right are mirrored in the image.
  const m = near ? 1 : -1;
  const P = (dx: number, dy: number, c = 0.9): Keypoint => [round1(fx + dx * m * s), round1(fy - dy * s + bob), c];
  const lift = swing * 0.75;
  return [
    P(0, 1.68), // nose
    P(-0.03, 1.71), P(0.03, 1.71), // eyes
    P(-0.07, 1.69), P(0.07, 1.69), // ears
    P(-0.2, 1.45), P(0.2, 1.45), // shoulders
    P(-0.28, 1.15), P(0.3, 1.2 + lift * 0.5), // elbows
    P(-0.3, 0.9), P(0.32, 1.0 + lift, 0.75), // wrists
    P(-0.13, 0.95), P(0.13, 0.95), // hips
    P(-0.17, 0.5), P(0.17, 0.5), // knees
    P(-0.2, 0.03), P(0.2, 0.03), // ankles
  ];
}

function hash(s: string): number {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 16777619);
  return h >>> 0;
}
const round1 = (v: number) => Math.round(v * 10) / 10;
const round2 = (v: number) => Math.round(v * 100) / 100;
