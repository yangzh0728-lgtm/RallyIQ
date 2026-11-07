import type { EventsResult, FootContact, FootworkEvent, FootworkKind, FootworkResult, OverlayData, PlayerId, ShotFootwork, SplitStepTiming } from "./types";

export const FOOTWORK_ZH: Record<FootworkKind, string> = {
  split_step: "启动步",
  lunge: "弓步",
  jump: "起跳",
  chasse: "并步",
  cross_step: "交叉步",
  run: "跑动",
  recovery: "回位",
};

/** One-character marks for the timeline. */
export const FOOTWORK_MARK: Record<FootworkKind, string> = {
  split_step: "启",
  lunge: "弓",
  jump: "跳",
  chasse: "并",
  cross_step: "交",
  run: "跑",
  recovery: "回",
};

export const SPLIT_ZH: Record<SplitStepTiming, string> = {
  on_time: "及时",
  early: "偏早",
  late: "偏晚",
  none: "没有",
  unknown: "无法判断",
};

/** Kinds shown on the timeline and the video by default (recovery and step patterns are in the panel). */
export const KEY_KINDS: FootworkKind[] = ["split_step", "lunge", "jump"];

const TRAIL_S = 0.8; // footprints stay visible this long after the foot lifts
const LABEL_S = 0.7; // event labels stay on the video this long

/** Fast lookup of the contacts and events around a frame. */
export class FootIndex {
  readonly contacts: FootContact[];
  readonly events: FootworkEvent[];
  private readonly starts: number[];
  private readonly maxLen: number;
  readonly byShot: Map<string, ShotFootwork>;

  constructor(readonly fw: FootworkResult) {
    this.contacts = [...fw.contacts].sort((a, b) => a.start_frame - b.start_frame);
    this.starts = this.contacts.map((c) => c.start_frame);
    this.maxLen = this.contacts.reduce((m, c) => Math.max(m, c.end_frame - c.start_frame), 0);
    this.events = [...fw.events].sort((a, b) => a.frame - b.frame);
    this.byShot = new Map(fw.shots.map((s) => [s.shot_id, s]));
  }

  /** Contacts touching [frame - trail, frame]: planted now or lifted within the trail. */
  near(frame: number, trailFrames: number): FootContact[] {
    const lo = lowerBound(this.starts, frame - trailFrames - this.maxLen);
    const out: FootContact[] = [];
    for (let i = lo; i < this.contacts.length && this.contacts[i].start_frame <= frame; i++) {
      const c = this.contacts[i];
      if (c.end_frame >= frame - trailFrames) out.push(c);
    }
    return out;
  }

  /** Events whose key frame lies in [frame - window, frame]. */
  recentEvents(frame: number, windowFrames: number, kinds: FootworkKind[] = KEY_KINDS): FootworkEvent[] {
    return this.events.filter((e) => e.frame <= frame && e.frame >= frame - windowFrames && kinds.includes(e.kind));
  }
}

function lowerBound(a: number[], x: number): number {
  let lo = 0;
  let hi = a.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (a[mid] < x) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

/** Footprints on the video: filled = planted now, outline = lifted a moment ago; plus recent action labels. */
export function drawFootwork(
  ctx: CanvasRenderingContext2D,
  idx: FootIndex,
  frame: number,
  fps: number,
  colorOf: (p: PlayerId | null) => string,
  px: number,
  opacity: number,
) {
  const trail = Math.round(TRAIL_S * fps);
  const lastImage = new Map<PlayerId, [number, number]>();
  for (const c of idx.near(frame, trail)) {
    const planted = c.end_frame >= frame;
    const age = planted ? 0 : (frame - c.end_frame) / trail;
    const color = colorOf(c.player_id);
    const [x, y] = c.image;
    lastImage.set(c.player_id, [x, y]);
    ctx.globalAlpha = opacity * (planted ? 0.95 : 0.6 * (1 - age));
    ctx.beginPath();
    ctx.ellipse(x, y, 12 * px, 6 * px, 0, 0, Math.PI * 2);
    ctx.lineWidth = 2 * px;
    ctx.strokeStyle = color;
    if (planted) {
      ctx.fillStyle = `${color}cc`;
      ctx.fill();
      ctx.strokeStyle = "#0b1220";
    }
    ctx.stroke();
    if (planted) {
      ctx.fillStyle = "#0b1220";
      ctx.font = `bold ${9 * px}px sans-serif`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(c.foot === "left" ? "L" : "R", x, y + 0.5 * px);
    }
  }
  // action labels (split step, lunge, jump) next to the player's feet
  for (const e of idx.recentEvents(frame, Math.round(LABEL_S * fps))) {
    const at = lastImage.get(e.player_id);
    if (!at) continue;
    const age = (frame - e.frame) / (LABEL_S * fps);
    ctx.globalAlpha = opacity * (1 - 0.7 * age);
    const text = FOOTWORK_ZH[e.kind] + (e.kind === "split_step" && e.value != null ? ` ${e.value > 0 ? "+" : ""}${Math.round(e.value)}ms` : "");
    ctx.font = `bold ${13 * px}px sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "bottom";
    const w = ctx.measureText(text).width + 8 * px;
    const ty = at[1] + 26 * px;
    ctx.fillStyle = "#0b1220d0";
    ctx.fillRect(at[0] - w / 2, ty - 16 * px, w, 18 * px);
    ctx.fillStyle = colorOf(e.player_id);
    ctx.fillText(text, at[0], ty);
  }
  ctx.globalAlpha = 1;
  ctx.textAlign = "start";
  ctx.textBaseline = "alphabetic";
}

export interface FootworkSummary {
  judged: number;
  splitPct: number | null;
  onTimePct: number | null;
  reactionMs: number | null;
  recoveredPct: number | null;
  lunges: number;
  jumps: number;
}

function median(xs: number[]): number | null {
  if (!xs.length) return null;
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

/** Per-player numbers for the panel, from the shot-linked footwork. */
export function summarize(fw: FootworkResult, pid: PlayerId): FootworkSummary {
  const shots = fw.shots.filter((s) => s.player_id === pid);
  const judged = shots.filter((s) => s.split_step !== "unknown");
  const split = judged.filter((s) => s.split_step !== "none");
  const onTime = judged.filter((s) => s.split_step === "on_time");
  const gaps = shots.map((s) => s.recovery_gap_m).filter((g): g is number => g != null);
  const pct = (a: number, b: number) => (b ? Math.round((100 * a) / b) : null);
  return {
    judged: judged.length,
    splitPct: pct(split.length, judged.length),
    onTimePct: pct(onTime.length, judged.length),
    reactionMs: median(shots.map((s) => s.reaction_ms).filter((r): r is number => r != null)),
    recoveredPct: pct(gaps.filter((g) => g <= fw.base_radius_m).length, gaps.length),
    lunges: fw.events.filter((e) => e.player_id === pid && e.kind === "lunge").length,
    jumps: fw.events.filter((e) => e.player_id === pid && e.kind === "jump").length,
  };
}

// ---------------------------------------------------------------- mock (?mock)

/** Plausible footwork for the synthetic demo: alternating contacts under the ankles, a split step
 * for every opponent hit, and a lunge / jump now and then. */
export function mockFootwork(overlay: OverlayData, result: EventsResult): FootworkResult {
  const fps = overlay.video.fps;
  const contacts: FootContact[] = [];
  const seg = Math.round(0.35 * fps);
  const byFrame = new Map(overlay.frames.map((f) => [f.frame, f]));
  for (const f of overlay.frames) {
    for (const p of f.players) {
      if (!p.player_id || p.keypoints.length < 17) continue;
      for (const [foot, k, phase] of [["left", 15, 0], ["right", 16, Math.round(seg / 2)]] as const) {
        if ((f.frame + phase) % seg !== 0) continue;
        const end = f.frame + Math.round(seg * 0.7);
        if (!byFrame.has(end)) continue;
        const [x, y] = p.keypoints[k];
        contacts.push({ player_id: p.player_id, foot, start_frame: f.frame, end_frame: end, t_ms: f.t_ms, end_ms: Math.round((end * 1000) / fps), image: [x, y + 4], court: p.foot_court ?? null });
      }
    }
  }
  const events: FootworkEvent[] = [];
  const shots: ShotFootwork[] = [];
  const sorted = [...result.shots].sort((a, b) => a.frame - b.frame);
  sorted.forEach((s, i) => {
    if (!s.hitter) return;
    const other = (s.hitter === "P1" ? "P2" : s.hitter === "P2" ? "P1" : null) as PlayerId | null;
    if (other) {
      const off = [120, 180, -260, 90, 420][i % 5];
      const land = s.frame + Math.round((off / 1000) * fps);
      events.push({ event_id: "", player_id: other, kind: "split_step", frame: land, t_ms: Math.round((land * 1000) / fps), start_frame: land - 6, end_frame: land, court: null, value: off, shot_id: s.shot_id, label: `启动步（对手击球后 ${off} ms 落地）`, confidence: 0.7 });
    }
    const kind: FootworkKind | null = i % 4 === 1 ? "lunge" : i % 7 === 3 ? "jump" : null;
    if (kind) events.push({ event_id: "", player_id: s.hitter, kind, frame: s.frame, t_ms: s.t_ms, start_frame: s.frame - 8, end_frame: s.frame, court: s.hitter_court ?? null, value: kind === "lunge" ? 1.3 : 0.3, shot_id: s.shot_id, label: kind === "lunge" ? "弓步击球（两脚相距 1.3 米）" : "起跳（腾空约 0.40 秒）", confidence: 0.7 });
    const split: SplitStepTiming = i === 0 ? "unknown" : (["on_time", "on_time", "early", "on_time", "late"] as const)[(i - 1) % 5];
    shots.push({
      shot_id: s.shot_id, player_id: s.hitter, t_ms: s.t_ms, split_step: split, split_offset_ms: split === "unknown" ? null : [120, 180, -260, 90, 420][(i - 1) % 5],
      reaction_ms: i === 0 ? null : 180 + (i % 4) * 60, steps: 2 + (i % 4), move_m: 1 + (i % 3) * 0.6, peak_speed_mps: 2.5, pattern: i % 2 ? ["chasse"] : ["cross_step"],
      finish: kind === "lunge" ? "lunge" : kind === "jump" ? "jump" : "planted", stance_m: kind === "lunge" ? 1.3 : 0.7, recovery_ms: 700, recovery_gap_m: 0.6 + (i % 3) * 0.5,
      summary: `启动步${SPLIT_ZH[split]}，${2 + (i % 4)} 步移动 ${(1 + (i % 3) * 0.6).toFixed(1)} 米${kind === "lunge" ? "，弓步击球" : ""}（演示数据）`, confidence: 0.7,
    });
  });
  events.sort((a, b) => a.frame - b.frame).forEach((e, i) => (e.event_id = `fw${i + 1}`));
  return {
    schema_version: "0.2.0", video_id: overlay.video.video_id, fingerprint: "mock", base: [3.05, 3.6], base_radius_m: 1,
    contacts, steps: [], events, shots, quality: [], caveats: ["演示数据：脚步为合成结果，不代表真实分析。"],
  };
}
