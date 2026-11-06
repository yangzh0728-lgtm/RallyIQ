// Wire types for the RallyIQ API. Mirrors rallyiq/contracts/models.py (schema 0.1.0) and the
// generated contracts/schemas/*.json. Conventions:
//   * frame = 0-based frame index, t_ms = its time in ms of the original video
//   * image coords: px of the original video, origin top-left
//   * court coords: metres, x across 0..6.10, y along 0..13.40, y = 0 is the baseline NEARER the camera
//   * nullable = unknown; never a fake default

export type Point = [number, number];
export type Keypoint = [number, number, number]; // [x, y, conf] px, COCO-17 order
export type BBox = [number, number, number, number]; // [x1, y1, x2, y2] px

/** README levels: ok 可用 / needs_review 待复核 / undeterminable 不可判定. */
export type Status = "ok" | "needs_review" | "undeterminable";
export type Source = "model" | "human";
export type PlayerId = "P1" | "P2";
export type Side = "near" | "far";

export interface VideoMeta {
  video_id: string;
  filename: string;
  fps: number;
  frame_count: number;
  width: number;
  height: number;
  duration_ms: number;
  codec?: string | null;
  rotation?: number;
  variable_frame_rate?: boolean;
}

export type StageName = "probe" | "calibration" | "players" | "shuttle" | "events" | "overlay";
export type JobState = "queued" | "running" | "succeeded" | "failed" | "cancelled";

export interface StageRun {
  name: StageName | string;
  state: "pending" | "running" | "succeeded" | "failed" | "skipped";
  progress: number; // 0..1
  started_at?: string | null;
  finished_at?: string | null;
  error?: string | null;
}

export interface Job {
  job_id: string;
  video_id: string;
  state: JobState;
  stages: StageRun[];
  created_at: string;
  started_at?: string | null;
  finished_at?: string | null;
  error?: string | null;
}

export interface Video {
  video_id: string;
  filename: string;
  size_bytes: number;
  created_at: string;
  meta: VideoMeta | null;
  latest_job: Job | null;
  artifacts: StageName[];
}

export interface OverlayPlayer {
  player_id: PlayerId | null;
  side?: Side | null;
  bbox: BBox;
  keypoints: Keypoint[];
  foot_court?: Point | null;
  interpolated?: boolean;
}

export interface OverlayShuttle {
  x: number;
  y: number;
  source: "detected" | "interpolated";
  confidence: number;
}

export interface OverlayFrame {
  frame: number;
  t_ms: number;
  players: OverlayPlayer[];
  shuttle: OverlayShuttle | null;
}

export interface CourtLines {
  start_frame: number;
  end_frame: number; // inclusive
  lines: [Point, Point][];
  status: Status;
}

export interface OverlayData {
  schema_version?: string;
  video: VideoMeta;
  court: CourtLines[];
  keypoint_format?: "coco17";
  /** Keypoint index pairs to draw as bones. */
  skeleton: [number, number][];
  frames: OverlayFrame[];
}

export interface ShotEvent {
  shot_id: string;
  rally_id: string | null;
  frame: number;
  t_ms: number;
  hitter: PlayerId | null;
  hitter_confidence?: number | null;
  shuttle_px?: Point | null;
  shuttle_court?: Point | null;
  hitter_court?: Point | null;
  shot_type?: string | null;
  target_zone?: string | null;
  confidence: number;
  status: Status;
  source: Source;
  reasons?: string[];
}

export type RallyEndReason = "shuttle_lost" | "shuttle_static" | "video_end" | "manual" | "unknown";

export interface Rally {
  rally_id: string;
  start_frame: number;
  end_frame: number;
  start_ms: number;
  end_ms: number;
  shot_ids: string[];
  end_reason: RallyEndReason;
  winner: PlayerId | null;
  confidence: number;
  status: Status;
  source: Source;
}

export interface EventsResult {
  schema_version?: string;
  video_id: string;
  rallies: Rally[];
  shots: ShotEvent[];
  /** Only on GET /events (effective). */
  corrections_applied?: number;
  corrections_skipped?: string[];
}

export type CorrectionOp =
  | { op: "add_shot"; payload: { frame: number; hitter?: PlayerId | null; shuttle_px?: Point | null } }
  | { op: "delete_shot"; payload: { shot_id: string } }
  | { op: "set_hitter"; payload: { shot_id: string; hitter: PlayerId | null } }
  | { op: "move_shot"; payload: { shot_id: string; frame: number } }
  | { op: "add_rally"; payload: { start_frame: number; end_frame: number } }
  | { op: "delete_rally"; payload: { rally_id: string } }
  | { op: "set_rally_bounds"; payload: { rally_id: string; start_frame?: number; end_frame?: number } }
  | { op: "set_rally_winner"; payload: { rally_id: string; winner: PlayerId | null } }
  | { op: "swap_player_ids"; payload: { from_frame: number } }
  | { op: "confirm_shot"; payload: { shot_id: string } }
  | { op: "confirm_rally"; payload: { rally_id: string } };

/** hard_example (default true on the backend): the fix also goes into the evaluation set. */
export type CorrectionIn = CorrectionOp & { note?: string | null; hard_example?: boolean };

export type Correction = CorrectionIn & {
  correction_id: string;
  video_id: string;
  created_at: string;
  seq: number;
};

export interface HeatmapData {
  video_id: string;
  player_id: PlayerId | null;
  cell_m: number;
  nx: number;
  ny: number;
  counts: number[][]; // counts[iy][ix]
  samples: number;
}

export const PLAYER_IDS: PlayerId[] = ["P1", "P2"];
export const PLAYER_COLORS: Record<PlayerId, string> = { P1: "#38bdf8", P2: "#f472b6" };
export const PLAYER_LABELS: Record<PlayerId, string> = { P1: "P1", P2: "P2" };
