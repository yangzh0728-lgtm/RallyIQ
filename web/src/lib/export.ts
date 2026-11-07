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
  "shot_type", "shot_type_confidence", "shot_type_status", "shot_type_source",
  "rally_no", "shot_no", "rally_shots", "rally_winner",
];

/** 1-based rally number (by start time) and 1-based shot number within its rally, as the backend writes them. */
export function rallyPositions(ev: EventsResult): { rallyNo: Map<string, number>; shotNo: Map<string, number> } {
  const rallies = [...ev.rallies].sort((a, b) => a.start_frame - b.start_frame);
  const frame = new Map(ev.shots.map((s) => [s.shot_id, s.frame]));
  const shotNo = new Map<string, number>();
  for (const r of rallies) {
    r.shot_ids
      .filter((id) => frame.has(id))
      .sort((a, b) => frame.get(a)! - frame.get(b)!)
      .forEach((id, k) => shotNo.set(id, k + 1));
  }
  return { rallyNo: new Map(rallies.map((r, i) => [r.rally_id, i + 1])), shotNo };
}

function cell(v: unknown): string {
  if (v === null || v === undefined) return "";
  const s = String(v);
  return /[",\r\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

/** One row per shot; unknown values stay empty (no fake defaults). */
export function exportCsv(ev: EventsResult): string {
  const { rallyNo, shotNo } = rallyPositions(ev);
  const rallyById = new Map(ev.rallies.map((r) => [r.rally_id, r]));
  const rows = ev.shots.map((s) => {
    const r = s.rally_id ? rallyById.get(s.rally_id) : undefined;
    const secs = s.t_ms / 1000;
    const time = `${String(Math.floor(secs / 60)).padStart(2, "0")}:${(secs % 60).toFixed(3).padStart(6, "0")}`;
    return [
      s.shot_id, s.rally_id, s.frame, s.t_ms, time, s.hitter, s.hitter ? TEAM_OF[s.hitter] : null, s.confidence.toFixed(3), s.status, s.source,
      ...(s.shuttle_px ?? [null, null]), ...(s.hitter_court ?? [null, null]),
      s.shot_type, s.stroke ? s.stroke.confidence.toFixed(3) : null, s.stroke?.status, s.stroke?.source,
      s.rally_id ? rallyNo.get(s.rally_id) : null, shotNo.get(s.shot_id), r?.shot_ids.length, r?.winner,
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
