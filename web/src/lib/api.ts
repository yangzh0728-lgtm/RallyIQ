import { applyCorrections } from "./corrections";
import { CORNERS_M, homographyFrom4, projectCourtLines } from "./court";
import { generateMock, type MockAnalysis } from "./mock";
import type { Correction, CorrectionIn, EventsResult, Job, OverlayData, Point, StageName, StageRun, Video } from "./types";

export interface PublicConfig {
  auth: boolean;
  signup: "invite" | "open" | "closed";
  direct_upload: boolean;
  max_upload_mb: number;
}

export interface User {
  user_id: string;
  email: string;
  is_admin: boolean;
}

/** Fired on any 401 so the app can show the login screen. */
export const UNAUTHORIZED_EVENT = "rallyiq:unauthorized";

export interface Api {
  readonly mode: "live" | "mock";
  getConfig(): Promise<PublicConfig>;
  /** Current account, or null when signed out. */
  me(): Promise<User | null>;
  login(email: string, password: string): Promise<User>;
  signup(email: string, password: string, inviteCode?: string): Promise<User>;
  logout(): Promise<void>;
  deleteVideo(id: string): Promise<void>;
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
  /** URL for the <video> element; null when there is no file (synthetic demo). */
  videoFileUrl(videoId: string): string | null;
  /** Server export URL, or null if the client builds the export itself (mock). */
  exportUrl(videoId: string, format: "json" | "csv"): string | null;
}

// ---------------------------------------------------------------- live (FastAPI, rallyiq/api/app.py)

const BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? "/api";
const enc = encodeURIComponent;

async function http<T>(path: string, init?: RequestInit & { quiet401?: boolean }): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    credentials: "include",
    ...init,
    headers: { Accept: "application/json", ...(init?.body ? { "Content-Type": "application/json" } : {}), ...init?.headers },
  });
  if (res.status === 401 && !init?.quiet401 && typeof window !== "undefined") {
    window.dispatchEvent(new Event(UNAUTHORIZED_EVENT));
  }
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
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

function xhrSend(
  method: string,
  url: string,
  body: XMLHttpRequestBodyInit,
  headers: Record<string, string>,
  onProgress?: (fraction: number) => void,
): Promise<string> {
  // XHR (not fetch) for upload progress on large phone videos.
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open(method, url);
    xhr.withCredentials = url.startsWith(BASE);
    for (const [k, v] of Object.entries(headers)) xhr.setRequestHeader(k, v);
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress?.(e.loaded / e.total);
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) return resolve(xhr.responseText);
      if (xhr.status === 401) window.dispatchEvent(new Event(UNAUTHORIZED_EVENT));
      let detail = xhr.statusText;
      try {
        detail = JSON.parse(xhr.responseText).detail ?? detail;
      } catch {
        /* not json */
      }
      reject(new Error(`${xhr.status} ${detail}`));
    };
    xhr.onerror = () => reject(new Error("上传失败：网络错误（后端是否在运行？）"));
    xhr.send(body);
  });
}

let configCache: Promise<PublicConfig> | null = null;
const getLiveConfig = () =>
  (configCache ??= http<PublicConfig>("/config").catch((e) => {
    configCache = null;
    throw e;
  }));

export const liveApi: Api = {
  mode: "live",
  getConfig: getLiveConfig,
  async me() {
    try {
      return await http<User>("/auth/me", { quiet401: true });
    } catch (e) {
      if ((e as Error).message.startsWith("401")) return null;
      throw e;
    }
  },
  login: (email, password) => http<User>("/auth/login", { method: "POST", body: JSON.stringify({ email, password }), quiet401: true }),
  signup: (email, password, inviteCode) =>
    http<User>("/auth/signup", { method: "POST", body: JSON.stringify({ email, password, invite_code: inviteCode || null }) }),
  logout: () => http<void>("/auth/logout", { method: "POST" }),
  deleteVideo: (id) => http<void>(`/videos/${enc(id)}`, { method: "DELETE" }),
  listVideos: () => http<Video[]>("/videos"),
  getVideo: (id) => http<Video>(`/videos/${enc(id)}`),
  async uploadVideo(file, onProgress) {
    const cfg = await getLiveConfig();
    if (file.size > cfg.max_upload_mb * 1024 * 1024) throw new Error(`视频超过 ${cfg.max_upload_mb} MB 上限`);
    if (cfg.direct_upload) {
      // Online: the browser PUTs straight to object storage, then tells the API it is done.
      const t = await http<{ video_id: string; url: string; headers: Record<string, string> }>("/uploads", {
        method: "POST",
        body: JSON.stringify({ filename: file.name, size_bytes: file.size, content_type: file.type || null }),
      });
      await xhrSend("PUT", t.url, file, t.headers, onProgress);
      return http<Video>(`/uploads/${enc(t.video_id)}/complete`, { method: "POST" });
    }
    const fd = new FormData();
    fd.append("file", file);
    return JSON.parse(await xhrSend("POST", `${BASE}/videos`, fd, {}, onProgress)) as Video;
  },
  getJob: (id) => http<Job>(`/jobs/${enc(id)}`),
  getOverlay: (id) => http<OverlayData>(`/videos/${enc(id)}/overlay`),
  getEvents: (id) => http<EventsResult>(`/videos/${enc(id)}/events`),
  getRawEvents: (id) => http<EventsResult>(`/videos/${enc(id)}/events?raw=true`),
  listCorrections: (id) => http<Correction[]>(`/videos/${enc(id)}/corrections`),
  postCorrection: (id, c) => http<Correction>(`/videos/${enc(id)}/corrections`, { method: "POST", body: JSON.stringify(c) }),
  setManualCalibration: (id, corners) =>
    http<Job>(`/videos/${enc(id)}/calibration/manual`, { method: "PUT", body: JSON.stringify({ corners }) }),
  videoFileUrl: (id) => `${BASE}/videos/${enc(id)}/file`,
  exportUrl: (id, format) => `${BASE}/videos/${enc(id)}/export?format=${format}`,
};

// ---------------------------------------------------------------- mock (?mock)

const STAGES: StageName[] = ["probe", "calibration", "players", "shuttle", "events", "overlay"];
const STAGE_MS = 900;
const LS_KEY = "rallyiq.mock.corrections.v2";

interface MockVideo {
  video: Omit<Video, "latest_job">;
  objectUrl: string | null;
  job: { id: string; startedAt: number; from: number };
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
  const iso = (t: number) => new Date(t).toISOString();
  const delay = <T,>(v: T, ms = 100) => new Promise<T>((r) => setTimeout(() => r(structuredClone(v)), ms));

  const demoAt = Date.now() - 3600_000;
  videos.set("demo", {
    video: { video_id: "demo", filename: "示例单打（合成数据）.mp4", size_bytes: 0, created_at: iso(demoAt), meta: null, artifacts: [] },
    objectUrl: null,
    job: { id: "job_demo_1", startedAt: 0, from: 0 },
  });

  const analysis = (id: string, dims?: { w: number; h: number; d: number }) => {
    let a = analyses.get(id);
    if (!a) {
      const v = videos.get(id)!.video;
      a = generateMock(id, { width: dims?.w, height: dims?.h, durationMs: dims?.d, filename: v.filename });
      analyses.set(id, a);
      v.meta = a.overlay.video;
    }
    return a;
  };
  analysis("demo");

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
    getConfig: async () => ({ auth: false, signup: "closed", direct_upload: false, max_upload_mb: 4096 }),
    me: async () => null,
    login: async () => {
      throw new Error("演示模式没有账号");
    },
    signup: async () => {
      throw new Error("演示模式没有账号");
    },
    logout: async () => {},
    deleteVideo: async (id) => {
      must(id);
      videos.delete(id);
    },
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
      a.overlay = { ...a.overlay, court: [{ start_frame: 0, end_frame: a.overlay.video.frame_count - 1, lines: projectCourtLines(H), status: "ok" }] };
      const n = Number(mv.job.id.split("_").pop()) + 1;
      mv.job = { id: `job_${id}_${n}`, startedAt: Date.now(), from: 1 };
      return delay(jobOf(mv));
    },
    videoFileUrl: (id) => videos.get(id)?.objectUrl ?? null,
    exportUrl: () => null,
  };
}

const useMock =
  import.meta.env.VITE_USE_MOCK === "1" || (typeof location !== "undefined" && new URLSearchParams(location.search).has("mock"));
export const api: Api = useMock ? createMockApi() : liveApi;
