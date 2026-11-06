import { applyCorrections } from "./corrections";
import { boundaryLinesM, CORNERS_M, homographyFrom4, projectCourtLines, projectLines } from "./court";
import { generateMock, type MockAnalysis } from "./mock";
import { mockCoach, mockStats } from "./report";
import type { CoachReport, Correction, CorrectionIn, EventsResult, Job, MatchReport, OverlayData, Point, StageName, StageRun, Video } from "./types";

/** Per-video singles/doubles choice; "auto" lets the players stage decide. */
export type MatchFormatSetting = "auto" | "singles" | "doubles";

export interface Api {
  readonly mode: "live" | "mock";
  listVideos(): Promise<Video[]>;
  getVideo(id: string): Promise<Video>;
  uploadVideo(file: File, onProgress?: (fraction: number) => void): Promise<Video>;
  getJob(id: string): Promise<Job>;
  getOverlay(videoId: string): Promise<OverlayData>;
  /** Effective events (raw prediction + corrections). */
  getEvents(videoId: string): Promise<EventsResult>;
  /** Raw model prediction, untouched by corrections. */
  getRawEvents(videoId: string): Promise<EventsResult>;
  listCorrections(videoId: string): Promise<Correction[]>;
  postCorrection(videoId: string, c: CorrectionIn): Promise<Correction>;
  /** Corners in px, order near-left, near-right, far-right, far-left. Re-runs analysis from calibration. */
  setManualCalibration(videoId: string, corners: Point[]): Promise<Job>;
  getMatchFormat(videoId: string): Promise<MatchFormatSetting>;
  /** Force singles/doubles (or back to auto). Re-runs analysis from the players stage. */
  setMatchFormat(videoId: string, format: MatchFormatSetting): Promise<Job>;
  /** URL for the <video> element; null when there is no file (synthetic demo). */
  videoFileUrl(videoId: string): string | null;
  /** Server export URL, or null if the client builds the export itself (mock). */
  exportUrl(videoId: string, format: "json" | "csv"): string | null;
  /** Post-match stats (from effective events) + the cached coach report. */
  getReport(videoId: string): Promise<MatchReport>;
  /** (Re)generate the AI coach comments; provider null = server default. Can take a minute. */
  generateCoach(videoId: string, provider?: string | null): Promise<CoachReport>;
  /** Report download URL, or null if the client renders it (mock). */
  reportExportUrl(videoId: string, format: "md" | "json"): string | null;
}

// ---------------------------------------------------------------- live (FastAPI, rallyiq/api/app.py)

const BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? "/api";
const enc = encodeURIComponent;

async function http<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    ...init,
    headers: { Accept: "application/json", ...(init?.body ? { "Content-Type": "application/json" } : {}), ...init?.headers },
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const j = await res.json();
      detail = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail ?? j);
    } catch {
      /* not json */
    }
    throw new Error(`${res.status} ${detail}`);
  }
  return (await res.json()) as T;
}

export const liveApi: Api = {
  mode: "live",
  listVideos: () => http<Video[]>("/videos"),
  getVideo: (id) => http<Video>(`/videos/${enc(id)}`),
  uploadVideo: (file, onProgress) =>
    new Promise((resolve, reject) => {
      // XHR (not fetch) for upload progress on large phone videos.
      const xhr = new XMLHttpRequest();
      xhr.open("POST", `${BASE}/videos`);
      xhr.upload.onprogress = (e) => e.lengthComputable && onProgress?.(e.loaded / e.total);
      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) return resolve(JSON.parse(xhr.responseText) as Video);
        let detail = xhr.statusText;
        try {
          detail = JSON.parse(xhr.responseText).detail ?? detail;
        } catch {
          /* not json */
        }
        reject(new Error(`${xhr.status} ${detail}`));
      };
      xhr.onerror = () => reject(new Error("上传失败：网络错误（后端是否在运行？）"));
      const fd = new FormData();
      fd.append("file", file);
      xhr.send(fd);
    }),
  getJob: (id) => http<Job>(`/jobs/${enc(id)}`),
  getOverlay: (id) => http<OverlayData>(`/videos/${enc(id)}/overlay`),
  getEvents: (id) => http<EventsResult>(`/videos/${enc(id)}/events`),
  getRawEvents: (id) => http<EventsResult>(`/videos/${enc(id)}/events?raw=true`),
  listCorrections: (id) => http<Correction[]>(`/videos/${enc(id)}/corrections`),
  postCorrection: (id, c) => http<Correction>(`/videos/${enc(id)}/corrections`, { method: "POST", body: JSON.stringify(c) }),
  setManualCalibration: (id, corners) =>
    http<Job>(`/videos/${enc(id)}/calibration/manual`, { method: "PUT", body: JSON.stringify({ corners }) }),
  getMatchFormat: async (id) => (await http<{ match_format: MatchFormatSetting }>(`/videos/${enc(id)}/match_format`)).match_format,
  setMatchFormat: (id, format) =>
    http<Job>(`/videos/${enc(id)}/match_format`, { method: "PUT", body: JSON.stringify({ match_format: format }) }),
  videoFileUrl: (id) => `${BASE}/videos/${enc(id)}/file`,
  exportUrl: (id, format) => `${BASE}/videos/${enc(id)}/export?format=${format}`,
  getReport: (id) => http<MatchReport>(`/videos/${enc(id)}/report`),
  generateCoach: (id, provider) =>
    http<CoachReport>(`/videos/${enc(id)}/report/coach`, { method: "POST", body: JSON.stringify({ provider: provider ?? null }) }),
  reportExportUrl: (id, format) => `${BASE}/videos/${enc(id)}/report/export?format=${format}`,
};

// ---------------------------------------------------------------- mock (?mock)

const STAGES: StageName[] = ["probe", "calibration", "players", "shuttle", "events", "overlay"];
const STAGE_MS = 900;
const LS_KEY = "rallyiq.mock.corrections.v2";

interface MockVideo {
  video: Omit<Video, "latest_job">;
  objectUrl: string | null;
  job: { id: string; startedAt: number; from: number };
  format?: MatchFormatSetting;
}

function loadCorrections(): Record<string, Correction[]> {
  try {
    return JSON.parse(localStorage.getItem(LS_KEY) ?? "{}");
  } catch {
    return {};
  }
}
function saveCorrections(all: Record<string, Correction[]>) {
  try {
    localStorage.setItem(LS_KEY, JSON.stringify(all));
  } catch {
    /* storage unavailable: keep in memory */
  }
}

function readVideoMeta(url: string): Promise<{ w: number; h: number; d: number }> {
  return new Promise((resolve) => {
    const v = document.createElement("video");
    v.preload = "metadata";
    v.onloadedmetadata = () => resolve({ w: v.videoWidth || 1280, h: v.videoHeight || 720, d: (v.duration || 75) * 1000 });
    v.onerror = () => resolve({ w: 1280, h: 720, d: 75_000 });
    v.src = url;
  });
}

export function createMockApi(): Api {
  const videos = new Map<string, MockVideo>();
  const analyses = new Map<string, MockAnalysis>();
  const corrections = loadCorrections();
  const coaches = new Map<string, CoachReport>();
  const iso = (t: number) => new Date(t).toISOString();
  const delay = <T,>(v: T, ms = 100) => new Promise<T>((r) => setTimeout(() => r(structuredClone(v)), ms));

  const demoAt = Date.now() - 3600_000;
  videos.set("demo", {
    video: { video_id: "demo", filename: "示例单打（合成数据）.mp4", size_bytes: 0, created_at: iso(demoAt), meta: null, artifacts: [] },
    objectUrl: null,
    job: { id: "job_demo_1", startedAt: 0, from: 0 },
  });
  videos.set("demo_doubles", {
    video: { video_id: "demo_doubles", filename: "示例双打（合成数据）.mp4", size_bytes: 0, created_at: iso(demoAt + 1000), meta: null, artifacts: [] },
    objectUrl: null,
    job: { id: "job_demo_doubles_1", startedAt: 0, from: 0 },
    format: "doubles",
  });

  const analysis = (id: string, dims?: { w: number; h: number; d: number }) => {
    let a = analyses.get(id);
    if (!a) {
      const mv = videos.get(id)!;
      const v = mv.video;
      a = generateMock(id, { width: dims?.w ?? v.meta?.width, height: dims?.h ?? v.meta?.height, durationMs: dims?.d ?? v.meta?.duration_ms, filename: v.filename, doubles: mv.format === "doubles" });
      analyses.set(id, a);
      v.meta = a.overlay.video;
    }
    return a;
  };
  analysis("demo");
  analysis("demo_doubles");

  const must = (id: string) => {
    const v = videos.get(id);
    if (!v) throw new Error(`404 video ${id} not found`);
    return v;
  };
  const jobOf = (mv: MockVideo): Job => {
    const elapsed = Date.now() - mv.job.startedAt;
    const stages: StageRun[] = STAGES.map((name, i) => {
      if (i < mv.job.from) return { name, state: "skipped", progress: 0 };
      const k = i - mv.job.from;
      const p = Math.min(1, Math.max(0, (elapsed - k * STAGE_MS) / STAGE_MS));
      return { name, state: p >= 1 ? "succeeded" : p > 0 ? "running" : "pending", progress: Math.round(p * 100) / 100 };
    });
    const done = stages.every((s) => s.state === "succeeded" || s.state === "skipped");
    return {
      job_id: mv.job.id,
      video_id: mv.video.video_id,
      state: done ? "succeeded" : stages.some((s) => s.state === "running") ? "running" : "queued",
      stages,
      created_at: iso(mv.job.startedAt),
      started_at: iso(mv.job.startedAt),
      finished_at: done ? iso(mv.job.startedAt + (STAGES.length - mv.job.from) * STAGE_MS) : null,
      error: null,
    };
  };
  const videoOf = (mv: MockVideo): Video => ({
    ...mv.video,
    latest_job: jobOf(mv),
    artifacts: jobOf(mv).state === "succeeded" ? STAGES : [],
  });
  const fps = (id: string) => analysis(id).overlay.video.fps;

  return {
    mode: "mock",
    listVideos: () => delay([...videos.values()].map(videoOf).sort((a, b) => b.created_at.localeCompare(a.created_at))),
    getVideo: async (id) => delay(videoOf(must(id))),
    async uploadVideo(file, onProgress) {
      for (let i = 1; i <= 10; i++) {
        await delay(null, 60);
        onProgress?.(i / 10);
      }
      const url = URL.createObjectURL(file);
      const dims = await readVideoMeta(url);
      const id = `vid_${Date.now().toString(36)}`;
      const now = Date.now();
      videos.set(id, {
        video: { video_id: id, filename: file.name, size_bytes: file.size, created_at: iso(now), meta: null, artifacts: [] },
        objectUrl: url,
        job: { id: `job_${id}_1`, startedAt: now, from: 0 },
      });
      analysis(id, dims);
      return videoOf(must(id));
    },
    async getJob(jobId) {
      const mv = [...videos.values()].find((v) => v.job.id === jobId);
      if (!mv) throw new Error("404 job not found");
      return delay(jobOf(mv));
    },
    getOverlay: async (id) => (must(id), delay(analysis(id).overlay, 200)),
    getEvents: async (id) => (must(id), delay(applyCorrections(analysis(id).events, corrections[id] ?? [], fps(id)))),
    getRawEvents: async (id) => (must(id), delay(analysis(id).events)),
    listCorrections: async (id) => (must(id), delay(corrections[id] ?? [])),
    async postCorrection(id, req) {
      must(id);
      const list = (corrections[id] ??= []);
      const seq = list.length + 1;
      const c = {
        ...req,
        hard_example: req.hard_example ?? true,
        note: req.note ?? null,
        correction_id: `cor_${seq}`,
        video_id: id,
        created_at: new Date().toISOString(),
        seq,
      } as Correction;
      list.push(c);
      saveCorrections(corrections);
      return delay(c);
    },
    async setManualCalibration(id, corners) {
      const mv = must(id);
      const H = homographyFrom4(CORNERS_M, corners);
      if (!H) throw new Error("422 四个角点无效（共线或重复）");
      const a = analysis(id);
      const boundary = projectLines(H, boundaryLinesM(a.overlay.match_format === "doubles" ? "doubles" : "singles"));
      a.overlay = { ...a.overlay, court: [{ start_frame: 0, end_frame: a.overlay.video.frame_count - 1, lines: projectCourtLines(H), boundary, status: "ok" }] };
      const n = Number(mv.job.id.split("_").pop()) + 1;
      mv.job = { id: `job_${id}_${n}`, startedAt: Date.now(), from: 1 };
      return delay(jobOf(mv));
    },
    getMatchFormat: async (id) => delay(must(id).format ?? "auto"),
    async setMatchFormat(id, format) {
      const mv = must(id);
      mv.format = format;
      analyses.delete(id); // regenerate with the new number of players
      analysis(id);
      const n = Number(mv.job.id.split("_").pop()) + 1;
      mv.job = { id: `job_${id}_${n}`, startedAt: Date.now(), from: STAGES.indexOf("players") };
      return delay(jobOf(mv));
    },
    videoFileUrl: (id) => videos.get(id)?.objectUrl ?? null,
    exportUrl: () => null,
    async getReport(id) {
      must(id);
      const stats = mockStats(applyCorrections(analysis(id).events, corrections[id] ?? [], fps(id)), analysis(id).overlay);
      const coach = coaches.get(id) ?? null;
      return delay({ stats, coach, coach_stale: !!coach && coach.stats_fingerprint !== stats.fingerprint });
    },
    async generateCoach(id) {
      const { stats } = await this.getReport(id);
      const c = mockCoach(stats);
      coaches.set(id, c);
      return delay(c, 600);
    },
    reportExportUrl: () => null,
  };
}

const useMock =
  import.meta.env.VITE_USE_MOCK === "1" || (typeof location !== "undefined" && new URLSearchParams(location.search).has("mock"));
export const api: Api = useMock ? createMockApi() : liveApi;
