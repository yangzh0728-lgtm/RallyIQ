import type { Correction, EventsResult, PlayerId, Rally, ShotEvent } from "./types";

const SWAP: Record<string, PlayerId | null> = { P1: "P2", P2: "P1" };

/**
 * Apply append-only corrections on top of raw events. Port of rallyiq/api/corrections.py so mock
 * mode behaves like the backend. Pure: raw events are never mutated.
 */
export function applyCorrections(raw: EventsResult, corrections: Correction[], fps: number): EventsResult {
  const tms = (frame: number) => Math.round((frame * 1000) / fps);
  const shots = new Map<string, ShotEvent>(raw.shots.map((s) => [s.shot_id, structuredClone(s)]));
  const rallies = new Map<string, Rally>(raw.rallies.map((r) => [r.rally_id, structuredClone(r)]));
  const skipped: string[] = [];

  for (const c of corrections) {
    switch (c.op) {
      case "add_shot": {
        const id = `h${String(c.seq).padStart(4, "0")}`;
        const f = c.payload.frame;
        shots.set(id, {
          shot_id: id, rally_id: null, frame: f, t_ms: tms(f), hitter: c.payload.hitter ?? null, hitter_confidence: null,
          shuttle_px: c.payload.shuttle_px ?? null, shuttle_court: null, hitter_court: null, shot_type: null, target_zone: null,
          confidence: 1, status: "ok", source: "human",
          reasons: [`added by ${c.correction_id}`],
        });
        break;
      }
      case "delete_shot":
      case "set_hitter":
      case "move_shot":
      case "confirm_shot": {
        const s = shots.get(c.payload.shot_id);
        if (!s) {
          skipped.push(c.correction_id);
          break;
        }
        if (c.op === "delete_shot") {
          shots.delete(s.shot_id);
          break;
        }
        if (c.op === "set_hitter") {
          s.hitter = c.payload.hitter;
          s.hitter_confidence = 1;
        } else if (c.op === "move_shot") {
          s.frame = c.payload.frame;
          s.t_ms = tms(c.payload.frame);
        }
        s.source = "human";
        s.status = "ok";
        break;
      }
      case "add_rally": {
        const id = `hr${String(c.seq).padStart(4, "0")}`;
        const { start_frame: sf, end_frame: ef } = c.payload;
        rallies.set(id, {
          rally_id: id, start_frame: sf, end_frame: ef, start_ms: tms(sf), end_ms: tms(ef), shot_ids: [],
          end_reason: "manual", winner: null, confidence: 1, status: "ok", source: "human",
        });
        break;
      }
      case "delete_rally":
      case "set_rally_bounds":
      case "set_rally_winner":
      case "confirm_rally": {
        const r = rallies.get(c.payload.rally_id);
        if (!r) {
          skipped.push(c.correction_id);
          break;
        }
        if (c.op === "delete_rally") {
          rallies.delete(r.rally_id);
          break;
        }
        if (c.op === "set_rally_winner") r.winner = c.payload.winner;
        else if (c.op === "set_rally_bounds") {
          if (c.payload.start_frame !== undefined) {
            r.start_frame = c.payload.start_frame;
            r.start_ms = tms(r.start_frame);
          }
          if (c.payload.end_frame !== undefined) {
            r.end_frame = c.payload.end_frame;
            r.end_ms = tms(r.end_frame);
            r.end_reason = "manual";
          }
        }
        r.source = "human";
        r.status = "ok";
        break;
      }
      case "swap_player_ids": {
        const f0 = c.payload.from_frame;
        for (const s of shots.values()) if (s.frame >= f0 && s.hitter) s.hitter = SWAP[s.hitter];
        for (const r of rallies.values()) if (r.start_frame >= f0 && r.winner) r.winner = SWAP[r.winner];
        break;
      }
    }
  }

  // Re-derive rally membership from frames so added/moved shots land in the right rally.
  const orderedRallies = [...rallies.values()].sort((a, b) => a.start_frame - b.start_frame);
  orderedRallies.forEach((r) => (r.shot_ids = []));
  const orderedShots = [...shots.values()].sort((a, b) => a.frame - b.frame);
  for (const s of orderedShots) {
    const r = orderedRallies.find((x) => x.start_frame <= s.frame && s.frame <= x.end_frame);
    s.rally_id = r?.rally_id ?? null;
    r?.shot_ids.push(s.shot_id);
  }
  return {
    ...raw,
    rallies: orderedRallies,
    shots: orderedShots,
    corrections_applied: corrections.length,
    corrections_skipped: skipped,
  };
}

export function rallyAt(rallies: Rally[], t: number): Rally | undefined {
  return rallies.find((r) => t >= r.start_ms && t <= r.end_ms);
}

export function describeCorrection(c: Correction, fps: number): string {
  const t = (frame: number) => formatMs((frame * 1000) / fps);
  const who = (p: PlayerId | null | undefined) => p ?? "未知";
  switch (c.op) {
    case "add_shot":
      return `新增击球 @ ${t(c.payload.frame)}（${who(c.payload.hitter)}）`;
    case "delete_shot":
      return `删除击球 ${c.payload.shot_id}`;
    case "set_hitter":
      return `击球 ${c.payload.shot_id} 击球者 → ${who(c.payload.hitter)}`;
    case "move_shot":
      return `击球 ${c.payload.shot_id} 移到 ${t(c.payload.frame)}`;
    case "add_rally":
      return `新增回合 ${t(c.payload.start_frame)}–${t(c.payload.end_frame)}`;
    case "delete_rally":
      return `删除回合 ${c.payload.rally_id}`;
    case "set_rally_bounds": {
      const { start_frame: s, end_frame: e } = c.payload;
      if (s === undefined && e === undefined) return `确认回合 ${c.payload.rally_id}`;
      return `回合 ${c.payload.rally_id} 边界 → ${s !== undefined ? t(s) : "…"}–${e !== undefined ? t(e) : "…"}`;
    }
    case "set_rally_winner":
      return `回合 ${c.payload.rally_id} 得分方 → ${who(c.payload.winner)}`;
    case "swap_player_ids":
      return `从 ${t(c.payload.from_frame)} 起交换 P1/P2`;
    case "confirm_shot":
      return `确认击球 ${c.payload.shot_id}`;
    case "confirm_rally":
      return `确认回合 ${c.payload.rally_id}`;
  }
}

export function formatMs(ms: number): string {
  const s = Math.max(0, ms) / 1000;
  const m = Math.floor(s / 60);
  const rest = (s - m * 60).toFixed(2).padStart(5, "0");
  return `${m}:${rest}`;
}
