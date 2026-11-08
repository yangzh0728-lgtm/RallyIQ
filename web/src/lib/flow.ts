// Rally log, score flow and side-by-side comparison for the report page.
// The backend computes the rally log (rallyiq/report/flow.py); ``rallyLog`` is the same rules for
// demo mode and for reports saved by older versions that do not carry ``stats.rallies``.

import type { EventsResult, GameScore, Metric, PlayerStats, RallyLine, ShotEvent } from "./types";
import { TEAM_OF } from "./types";

const GAME_POINTS = 21;
const GAME_CAP = 30;
const MIN_INTERVAL_S = 0.15;
const MAX_INTERVAL_S = 3.0;

export function sidesOf(format: EventsResult["match_format"]): [string, string] {
  return format === "doubles" ? ["A", "B"] : ["P1", "P2"];
}

function sideFn(ev: EventsResult) {
  const doubles = ev.match_format === "doubles";
  return (x: string | null | undefined): string | null => (!x ? null : doubles ? (TEAM_OF as Record<string, string>)[x] ?? x : x);
}

const gameOver = (score: Record<string, number>) => {
  const v = Object.values(score);
  const hi = Math.max(...v);
  const lo = Math.min(...v);
  return hi >= GAME_CAP || (hi >= GAME_POINTS && hi - lo >= 2);
};

export function rallyLog(ev: EventsResult, label: (t: string) => string = (t) => t): { rallies: RallyLine[]; games: GameScore[] } {
  const side = sideFn(ev);
  const sides = sidesOf(ev.match_format);
  const byId = new Map(ev.shots.map((s) => [s.shot_id, s]));
  const zero = () => Object.fromEntries(sides.map((s) => [s, 0])) as Record<string, number>;
  let score = zero();
  let gameNo = 1;
  let gameRallies = 0;
  const games: GameScore[] = [];
  const rallies = [...ev.rallies]
    .sort((a, b) => a.start_frame - b.start_frame)
    .map((r, i): RallyLine => {
      const ss = r.shot_ids.map((id) => byId.get(id)).filter((s): s is ShotEvent => !!s).sort((a, b) => a.frame - b.frame);
      const iv: number[] = [];
      for (let k = 1; k < ss.length; k++) {
        const a = side(ss[k - 1].hitter);
        const b = side(ss[k].hitter);
        const dt = (ss[k].t_ms - ss[k - 1].t_ms) / 1000;
        if (a && b && a !== b && dt >= MIN_INTERVAL_S && dt <= MAX_INTERVAL_S) iv.push(dt);
      }
      const winner = side(r.winner) as RallyLine["winner"];
      let lineScore: Record<string, number> | null = null;
      let lineGame: number | null = null;
      if (winner && winner in score) {
        score[winner]++;
        gameRallies++;
        lineScore = { ...score };
        lineGame = gameNo;
        if (gameOver(score)) {
          games.push({ game_no: gameNo, score: { ...score }, winner, rallies: gameRallies });
          score = zero();
          gameNo++;
          gameRallies = 0;
        }
      }
      const last = ss[ss.length - 1];
      return {
        rally_no: i + 1, rally_id: r.rally_id, start_ms: r.start_ms, end_ms: r.end_ms, shots: r.shot_ids.length,
        duration_s: Math.round((r.end_ms - r.start_ms) / 10) / 100, server: (ss[0] ? side(ss[0].hitter) : null) as RallyLine["server"],
        winner, last_hitter: last?.hitter ?? null, last_shot_type: last?.shot_type ? label(last.shot_type) : null,
        mean_interval_s: iv.length ? Math.round((100 * iv.reduce((a, b) => a + b, 0)) / iv.length) / 100 : null,
        game_no: lineGame, score: lineScore, status: r.status,
      };
    });
  if (gameRallies) games.push({ game_no: gameNo, score: { ...score }, winner: null, rallies: gameRallies });
  return { rallies, games };
}

/** Lead after each decided rally: positive = first side ahead. */
export function leadSeries(rallies: RallyLine[], sides: [string, string]) {
  return rallies
    .filter((r) => r.score && r.winner)
    .map((r) => ({ rally: r, lead: (r.score![sides[0]] ?? 0) - (r.score![sides[1]] ?? 0) }));
}

export interface CompareRow {
  key: string; // metric key without the player prefix, e.g. "win.rate_pct"
  label: string;
  unit: string | null;
  values: (Metric | null)[];
}

/** Rows of metrics every player has (by key suffix), in the first player's order; skips rows with no value. */
export function compareRows(players: PlayerStats[]): CompareRow[] {
  if (players.length < 2) return [];
  const suffix = (k: string) => k.slice(k.indexOf(".") + 1);
  const maps = players.map((p) => new Map(p.metrics.map((m) => [suffix(m.key), m])));
  return players[0].metrics
    .map((m) => {
      const key = suffix(m.key);
      return { key, label: m.label, unit: m.unit, values: maps.map((mp) => mp.get(key) ?? null) };
    })
    .filter((r) => r.values.some((v) => v && v.value !== null));
}
