// Wire types for the RallyIQ API. Mirrors rallyiq/contracts/models.py (schema 0.2.0) and the
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
/** Singles: P1/P2. Doubles: team A = P1 + P3, team B = P2 + P4. */
export type PlayerId = "P1" | "P2" | "P3" | "P4";
export type TeamId = "A" | "B";
export type MatchFormat = "singles" | "doubles";
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

export type StageName = "probe" | "calibration" | "players" | "shuttle" | "events" | "strokes" | "overlay";
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
  team?: TeamId | null;
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
  /** In-play rally boundary for the match format (singles or doubles sidelines). */
  boundary?: [Point, Point][];
  status: Status;
  /** Outer corners of the analysed court: near-left, near-right, far-right, far-left. */
  image_corners?: Point[] | null;
  /** Other courts in view (ignored by the analysis); the user can switch to one. */
  other_courts?: Point[][];
}

export interface OverlayData {
  schema_version?: string;
  video: VideoMeta;
  match_format?: MatchFormat;
  court: CourtLines[];
  keypoint_format?: "coco17";
  /** Keypoint index pairs to draw as bones. */
  skeleton: [number, number][];
  frames: OverlayFrame[];
}

/** Stroke types (击球类型): the merged ShuttleSet taxonomy without the Top_/Bottom_ prefix. */
export type ShotType =
  | "short_serve"
  | "long_serve"
  | "clear"
  | "smash"
  | "drop"
  | "net_shot"
  | "cross_net"
  | "push"
  | "net_kill"
  | "drive"
  | "lift"
  | "block";

export interface ShotTypeScore {
  shot_type: ShotType;
  p: number;
}

/** Stroke-type prediction (stage `strokes`) or human label (`set_shot_type`). */
export interface StrokeInfo {
  shot_type: ShotType | null; // null = undeterminable
  confidence: number;
  status: Status;
  source: Source;
  classifier?: string | null;
  candidates?: ShotTypeScore[];
  reasons?: string[];
  evidence?: Record<string, number | null>;
}

export interface ShotStroke {
  shot_id: string;
  frame: number;
  stroke: StrokeInfo;
}

export interface StrokesResult {
  schema_version?: string;
  video_id: string;
  classifier: string | null;
  shots: ShotStroke[];
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
  shot_type?: ShotType | null;
  stroke?: StrokeInfo | null;
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
  /** Singles: P1/P2. Doubles: team A/B. */
  winner: PlayerId | TeamId | null;
  confidence: number;
  status: Status;
  source: Source;
}

export interface EventsResult {
  schema_version?: string;
  video_id: string;
  match_format?: MatchFormat;
  rallies: Rally[];
  shots: ShotEvent[];
  /** Only on GET /events (effective). */
  corrections_applied?: number;
  corrections_skipped?: string[];
}

export type CorrectionOp =
  | { op: "add_shot"; payload: { frame: number; hitter?: PlayerId | null; shuttle_px?: Point | null; shot_type?: ShotType | null } }
  | { op: "delete_shot"; payload: { shot_id: string } }
  | { op: "set_hitter"; payload: { shot_id: string; hitter: PlayerId | null } }
  | { op: "move_shot"; payload: { shot_id: string; frame: number } }
  | { op: "add_rally"; payload: { start_frame: number; end_frame: number } }
  | { op: "delete_rally"; payload: { rally_id: string } }
  | { op: "set_rally_bounds"; payload: { rally_id: string; start_frame?: number; end_frame?: number } }
  | { op: "set_rally_winner"; payload: { rally_id: string; winner: PlayerId | TeamId | null } }
  /** Swap ids a/b (default P1/P2) from this frame on. */
  | { op: "swap_player_ids"; payload: { from_frame: number; a?: PlayerId; b?: PlayerId } }
  | { op: "confirm_shot"; payload: { shot_id: string } }
  | { op: "confirm_rally"; payload: { rally_id: string } }
  | { op: "set_shot_type"; payload: { shot_id: string; shot_type: ShotType | null } };

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
  team?: TeamId | null;
  cell_m: number;
  nx: number;
  ny: number;
  counts: number[][]; // counts[iy][ix]
  samples: number;
}

// ------------------------------------------------------------------ report (rallyiq/contracts/models.py)

export type MetricConfidence = "high" | "medium" | "low";

export interface Metric {
  key: string;
  label: string;
  value: number | null;
  unit: string | null;
  n: number;
  status: Status;
  confidence: MetricConfidence;
  note: string | null;
}

export interface Distribution {
  key: string;
  label: string;
  counts: Record<string, number>;
  n: number;
  status: Status;
  confidence: MetricConfidence;
  note: string | null;
}

export interface PlayerStats {
  player_id: PlayerId;
  metrics: Metric[];
  distributions: Distribution[];
}

export interface MatchStats {
  schema_version: string;
  video_id: string;
  fingerprint: string;
  duration_ms: number | null;
  calibration_status: Status | null;
  shot_types_available: boolean;
  metrics: Metric[];
  distributions: Distribution[];
  players: PlayerStats[];
  caveats: string[];
}

export type InsightKind = "strength" | "weakness" | "pattern" | "tactic" | "data_quality";

export interface CoachInsight {
  kind: InsightKind;
  player_id: PlayerId | null;
  title: string;
  detail: string;
  evidence: string[];
  confidence: MetricConfidence;
  grounded: boolean;
  warnings: string[];
}

export interface TrainingItem {
  player_id: PlayerId | null;
  drill: string;
  why: string;
  evidence: string[];
  retest_metric: string | null;
}

export interface CoachReport {
  schema_version: string;
  video_id: string;
  stats_fingerprint: string;
  provider: string;
  model: string | null;
  created_at: string;
  language: "zh";
  summary: string;
  insights: CoachInsight[];
  training: TrainingItem[];
  caveats: string[];
  fallback_reason: string | null;
  dropped: number;
}

export interface MatchReport {
  stats: MatchStats;
  coach: CoachReport | null;
  coach_stale: boolean;
}

export const PLAYER_IDS: PlayerId[] = ["P1", "P2"];
export const DOUBLES_PLAYER_IDS: PlayerId[] = ["P1", "P3", "P2", "P4"];
export const TEAM_IDS: TeamId[] = ["A", "B"];
export const TEAM_OF: Record<PlayerId, TeamId> = { P1: "A", P3: "A", P2: "B", P4: "B" };
export const TEAM_MEMBERS: Record<TeamId, PlayerId[]> = { A: ["P1", "P3"], B: ["P2", "P4"] };
// Team A in cool colours, team B in warm ones, so partners read as a pair.
export const PLAYER_COLORS: Record<PlayerId, string> = { P1: "#38bdf8", P3: "#a78bfa", P2: "#f472b6", P4: "#fb923c" };
export const PLAYER_LABELS: Record<PlayerId, string> = { P1: "P1", P2: "P2", P3: "P3", P4: "P4" };
export const TEAM_COLORS: Record<TeamId, string> = { A: "#38bdf8", B: "#f472b6" };
export const TEAM_LABELS: Record<TeamId, string> = { A: "A 队（P1+P3）", B: "B 队（P2+P4）" };

export const playerIdsFor = (format: MatchFormat | undefined): PlayerId[] => (format === "doubles" ? DOUBLES_PLAYER_IDS : PLAYER_IDS);

// ------------------------------------------------------------------ input quality (GET /videos/{id}/quality)

export type CameraView = "end_center" | "end_oblique" | "side" | "unknown";
export type CameraHeight = "high" | "medium" | "low" | "unknown";

export interface QualityCheck {
  key: string;
  label: string;
  value: string | null;
  score?: number | null;
  status: Status;
  note?: string | null;
}

export interface CapabilityReliability {
  capability: "court_metrics" | "players" | "shuttle" | "rallies" | "hits";
  label: string;
  status: Status;
  reasons: string[];
}

export interface VideoQuality {
  schema_version: string;
  video_id: string;
  fingerprint: string;
  camera_view: CameraView;
  camera_height: CameraHeight;
  checks: QualityCheck[];
  capabilities: CapabilityReliability[];
  summary: string;
}

// ------------------------------------------------------------------ hard examples (GET /hard_examples/summary)

export interface ConditionGroup {
  condition: string;
  value: string;
  videos: number;
  model_shots: number;
  model_rallies: number;
  errors: Record<string, number>;
  confirmed: number;
  hit_error_rate: number | null;
}

export interface HardExampleSummary {
  total: number;
  errors: number;
  videos: number;
  by_kind: Record<string, number>;
  groups: ConditionGroup[];
  weakest: ConditionGroup[];
}
