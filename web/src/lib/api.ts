import { applyCorrections } from "./corrections";
import { boundaryLinesM, CORNERS_M, homographyFrom4, projectCourtLines, projectLines } from "./court";
import { generateMock, type MockAnalysis } from "./mock";
import { mockQuality } from "./quality";
import { mockCoach, mockStats } from "./report";
import type { CoachReport, Correction, CorrectionIn, EventsResult, Job, MatchReport, OverlayData, Point, StageName, StageRun, Video, VideoQuality, HardExampleSummary, OpponentReport, PeopleAssignment, PersonSummary, PlayerProfile } from "./types";

/** Per-video singles/doubles choice; "auto" lets the players stage decide. */
export type MatchFormatSetting = "auto" | "singles" | "doubles";

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
  /** Start (or retry) the full analysis of a video. */
  startAnalysis(videoId: string): Promise<Job>;
  getOverlay(videoId: string): Promise<OverlayData>;
  /** Effective events (raw prediction + corrections). */
  getEvents(videoId: string): Promise<EventsResult>;
  /** Raw model prediction, untouched by corrections. */
  getRawEvents(videoId: string): Promise<EventsResult>;
  listCorrections(videoId: string): Promise<Correction[]>;
  postCorrection(videoId: string, c: CorrectionIn): Promise<Correction>;
  /** Corners in px, order near-left, near-right, far-right, far-left. Re-runs analysis from calibration. */
  /** ``otherCourts``: courts in view to keep ignoring (e.g. the one analysed before switching). */
  setManualCalibration(videoId: string, corners: Point[], otherCourts?: Point[][]): Promise<Job>;
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
  reportExportUrl(videoId: string, format: "md" | "json" | "html"): string | null;
  /** Input assessment: camera view, picture, visibility and how far each output can be trusted. */
  getQuality(videoId: string): Promise<VideoQuality>;
  /** MP4 clip of one rally (padded 1 s), or null when there is no source file (mock). */
  clipUrl(videoId: string, rallyId: string): string | null;
  /** Who P1..P4 are in this video, and which one is the user. */
  getPeople(videoId: string): Promise<PeopleAssignment>;
  setPeople(videoId: string, people: PeopleAssignment["people"], selfPlayer: PeopleAssignment["self_player"]): Promise<PeopleAssignment>;
  /** Append this reviewed match to the long-term log (one event per named player). */
  commitMemory(videoId: string): Promise<unknown[]>;
  /** Exclude this video from profiles from now on (appends a withdrawal). */
  withdrawMemory(videoId: string): Promise<unknown[]>;
  listPeople(): Promise<PersonSummary[]>;
  /** The user's profile, or ``person``'s. */
  getProfile(person?: string | null): Promise<PlayerProfile>;
  getOpponent(person: string): Promise<OpponentReport>;
  /** Corrections kept as hard examples, grouped by video condition. */
  getHardExampleSummary(): Promise<HardExampleSummary>;
  /** JSONL download of all hard examples, or null in mock mode. */
  hardExamplesUrl(): string | null;
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
  startAnalysis: (id) => http<Job>(`/videos/${enc(id)}/jobs`, { method: "POST", body: JSON.stringify({ from_stage: "probe" }) }),
  getOverlay: (id) => http<OverlayData>(`/videos/${enc(id)}/overlay`),
  getEvents: (id) => http<EventsResult>(`/videos/${enc(id)}/events`),
  getRawEvents: (id) => http<EventsResult>(`/videos/${enc(id)}/events?raw=true`),
  listCorrections: (id) => http<Correction[]>(`/videos/${enc(id)}/corrections`),
  postCorrection: (id, c) => http<Correction>(`/videos/${enc(id)}/corrections`, { method: "POST", body: JSON.stringify(c) }),
  setManualCalibration: (id, corners, otherCourts = []) =>
    http<Job>(`/videos/${enc(id)}/calibration/manual`, {
      method: "PUT",
      body: JSON.stringify({ corners, other_courts: otherCourts }),
    }),
  getMatchFormat: async (id) => (await http<{ match_format: MatchFormatSetting }>(`/videos/${enc(id)}/match_format`)).match_format,
  setMatchFormat: (id, format) =>
    http<Job>(`/videos/${enc(id)}/match_format`, { method: "PUT", body: JSON.stringify({ match_format: format }) }),
  videoFileUrl: (id) => `${BASE}/videos/${enc(id)}/file`,
  exportUrl: (id, format) => `${BASE}/videos/${enc(id)}/export?format=${format}`,
  getReport: (id) => http<MatchReport>(`/videos/${enc(id)}/report`),
  generateCoach: (id, provider) =>
    http<CoachReport>(`/videos/${enc(id)}/report/coach`, { method: "POST", body: JSON.stringify({ provider: provider ?? null }) }),
  reportExportUrl: (id, format) => `${BASE}/videos/${enc(id)}/report/export?format=${format}`,
  getQuality: (id) => http<VideoQuality>(`/videos/${enc(id)}/quality`),
  getPeople: (id) => http<PeopleAssignment>(`/videos/${enc(id)}/people`),
  setPeople: (id, people, self_player) =>
    http<PeopleAssignment>(`/videos/${enc(id)}/people`, { method: "PUT", body: JSON.stringify({ people, self_player }) }),
  commitMemory: (id) => http<unknown[]>(`/videos/${enc(id)}/memory/commit`, { method: "POST" }),
  withdrawMemory: (id) => http<unknown[]>(`/videos/${enc(id)}/memory/withdraw`, { method: "POST" }),
  listPeople: () => http<PersonSummary[]>("/people"),
  getProfile: (person) => http<PlayerProfile>(`/profile${person ? `?person=${enc(person)}` : ""}`),
  getOpponent: (person) => http<OpponentReport>(`/opponents/${enc(person)}`),
  clipUrl: (id, rallyId) => `${BASE}/videos/${enc(id)}/clip?rally_id=${enc(rallyId)}`,
  getHardExampleSummary: () => http<HardExampleSummary>("/hard_examples/summary"),
  hardExamplesUrl: () => `${BASE}/hard_examples?format=jsonl`,
};

// ---------------------------------------------------------------- mock (?mock)

const STAGES: StageName[] = ["probe", "calibration", "players", "shuttle", "events", "strokes", "overlay"];
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
    async startAnalysis(id) {
      const mv = must(id);
      const n = Number(mv.job.id.split("_").pop()) + 1;
      mv.job = { id: `job_${id}_${n}`, startedAt: Date.now(), from: 0 };
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
    async setManualCalibration(id, corners, otherCourts = []) {
      const mv = must(id);
      const H = homographyFrom4(CORNERS_M, corners);
      if (!H) throw new Error("422 四个角点无效（共线或重复）");
      const a = analysis(id);
      a.overlay = {
        ...a.overlay,
        court: [{
          start_frame: 0, end_frame: a.overlay.video.frame_count - 1, lines: projectCourtLines(H), status: "ok",
          boundary: projectLines(H, boundaryLinesM(a.overlay.match_format === "doubles" ? "doubles" : "singles")),
          image_corners: corners, other_courts: otherCourts,
        }],
      };
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
    async getQuality(id) {
      must(id);
      return delay(mockQuality(analysis(id).overlay));
    },
    async getHardExampleSummary() {
      const all = Object.values(corrections).flat().filter((c) => c.hard_example);
      const by_kind: Record<string, number> = {};
      for (const c of all) by_kind[c.op] = (by_kind[c.op] ?? 0) + 1;
      const errors = all.filter((c) => !c.op.startsWith("confirm") && c.op !== "set_rally_winner").length;
      return delay({ total: all.length, errors, videos: Object.values(corrections).filter((cs) => cs.some((c) => c.hard_example)).length, by_kind, groups: [], weakest: [] });
    },
    hardExamplesUrl: () => null,
    clipUrl: () => null,
    ...mockMemory(),
  };
}

const useMock =
  import.meta.env.VITE_USE_MOCK === "1" || (typeof location !== "undefined" && new URLSearchParams(location.search).has("mock"));
export const api: Api = useMock ? createMockApi() : liveApi;

/** Demo mode has no backend log: people can be named, but profiles need the real service. */
function mockMemory(): Pick<Api, "getPeople" | "setPeople" | "commitMemory" | "withdrawMemory" | "listPeople" | "getProfile" | "getOpponent"> {
  const people = new Map<string, PeopleAssignment>();
  const unavailable = () => Promise.reject(new Error("演示模式没有个人档案：连接后端后，确认过的比赛才会写入只追加的档案日志"));
  return {
    getPeople: async (id) => people.get(id) ?? { video_id: id, people: {}, self_player: null },
    setPeople: async (id, p, self_player) => {
      const a = { video_id: id, people: p, self_player };
      people.set(id, a);
      return a;
    },
    commitMemory: unavailable,
    withdrawMemory: unavailable,
    listPeople: async () => [],
    getProfile: unavailable,
    getOpponent: unavailable,
  };
}
