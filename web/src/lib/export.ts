import { TEAM_OF, type Correction, type EventsResult, type Video } from "./types";

// Client-side export for mock mode. Same shapes as GET /api/videos/{id}/export on the backend.

export function exportJson(video: Video, effective: EventsResult, raw: EventsResult, corrections: Correction[]): string {
  const { latest_job: _job, ...v } = video;
  void _job;
  return JSON.stringify(
    { video: v, events: effective, raw_events: raw, corrections, corrections_skipped: effective.corrections_skipped ?? [] },
    null,
    1,
  );
}

export const CSV_HEADER = [
  "shot_id", "rally_id", "frame", "t_ms", "time", "hitter", "team", "confidence", "status", "source",
  "shuttle_x", "shuttle_y", "hitter_court_x", "hitter_court_y",
];

function cell(v: unknown): string {
  if (v === null || v === undefined) return "";
  const s = String(v);
  return /[",\r\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

/** One row per shot; unknown values stay empty (no fake defaults). */
export function exportCsv(ev: EventsResult): string {
  const rows = ev.shots.map((s) => {
    const secs = s.t_ms / 1000;
    const time = `${String(Math.floor(secs / 60)).padStart(2, "0")}:${(secs % 60).toFixed(3).padStart(6, "0")}`;
    return [
      s.shot_id, s.rally_id, s.frame, s.t_ms, time, s.hitter, s.hitter ? TEAM_OF[s.hitter] : null, s.confidence.toFixed(3), s.status, s.source,
      ...(s.shuttle_px ?? [null, null]), ...(s.hitter_court ?? [null, null]),
    ];
  });
  return [CSV_HEADER, ...rows].map((r) => r.map(cell).join(",")).join("\r\n") + "\r\n";
}

export function download(filename: string, text: string, mime: string) {
  const url = URL.createObjectURL(new Blob([text], { type: mime }));
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
