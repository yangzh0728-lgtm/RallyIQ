import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { applyCorrections } from "./corrections";
import { applyH, CORNERS_M, homographyFrom4, invertH } from "./court";
import { CSV_HEADER, exportCsv } from "./export";
import { isVideoFile } from "./files";
import { computeHeat } from "./heatmap";
import { generateMock } from "./mock";
import { frameIndexAt } from "./overlay";
import { jobProgress } from "./players";
import { retypeDetail } from "./strokes";
import { mergeStrokes, strokeStats } from "./strokes";
import type { Correction, CorrectionOp, EventsResult } from "./types";

let seq = 0;
const corr = (op: CorrectionOp): Correction =>
  ({ ...op, hard_example: true, note: null, correction_id: `cor_${++seq}`, video_id: "v", created_at: "2026-01-01T00:00:00Z", seq }) as Correction;

const FPS = 30;
const base: EventsResult = {
  video_id: "v",
  rallies: [
    { rally_id: "r1", start_frame: 30, end_frame: 150, start_ms: 1000, end_ms: 5000, shot_ids: ["s1", "s2"], end_reason: "shuttle_lost", winner: null, confidence: 0.7, status: "needs_review", source: "model" },
    { rally_id: "r2", start_frame: 240, end_frame: 360, start_ms: 8000, end_ms: 12000, shot_ids: ["s3"], end_reason: "shuttle_lost", winner: "P1", confidence: 0.9, status: "ok", source: "model" },
  ],
  shots: [
    { shot_id: "s1", rally_id: "r1", frame: 45, t_ms: 1500, hitter: "P1", confidence: 0.9, status: "ok", source: "model" },
    { shot_id: "s2", rally_id: "r1", frame: 75, t_ms: 2500, hitter: "P1", confidence: 0.4, status: "needs_review", source: "model" },
    { shot_id: "s3", rally_id: "r2", frame: 270, t_ms: 9000, hitter: "P2", confidence: 0.8, status: "ok", source: "model" },
  ],
};

describe("homography", () => {
  it("maps the four corners exactly and inverts", () => {
    const img: [number, number][] = [[210, 660], [1070, 660], [810, 175], [470, 175]];
    const H = homographyFrom4(CORNERS_M, img)!;
    CORNERS_M.forEach((c, i) => {
      const [x, y] = applyH(H, c);
      expect(x).toBeCloseTo(img[i][0], 6);
      expect(y).toBeCloseTo(img[i][1], 6);
    });
    const [u, v] = applyH(invertH(H)!, applyH(H, [3, 6.7]));
    expect(u).toBeCloseTo(3, 6);
    expect(v).toBeCloseTo(6.7, 6);
  });
  it("rejects degenerate points", () => {
    expect(homographyFrom4(CORNERS_M, [[0, 0], [1, 1], [2, 2], [3, 3]])).toBeNull();
  });
});

describe("applyCorrections", () => {
  it("never mutates raw events", () => {
    const snapshot = JSON.stringify(base);
    applyCorrections(base, [corr({ op: "delete_shot", payload: { shot_id: "s1" } }), corr({ op: "set_hitter", payload: { shot_id: "s2", hitter: "P2" } })], FPS);
    expect(JSON.stringify(base)).toBe(snapshot);
  });

  it("adds, deletes, re-assigns, moves and confirms shots", () => {
    seq = 0;
    const out = applyCorrections(
      base,
      [
        corr({ op: "delete_shot", payload: { shot_id: "s2" } }),
        corr({ op: "set_hitter", payload: { shot_id: "s1", hitter: "P2" } }),
        corr({ op: "add_shot", payload: { frame: 90, hitter: "P1" } }),
        corr({ op: "move_shot", payload: { shot_id: "s3", frame: 273 } }),
        corr({ op: "confirm_shot", payload: { shot_id: "nope" } }),
      ],
      FPS,
    );
    expect(out.shots.map((s) => s.shot_id)).toEqual(["s1", "h0003", "s3"]);
    expect(out.shots[0]).toMatchObject({ hitter: "P2", status: "ok", source: "human" });
    expect(out.shots[1]).toMatchObject({ rally_id: "r1", source: "human", t_ms: 3000 });
    expect(out.shots[2]).toMatchObject({ frame: 273, t_ms: 9100 });
    expect(out.rallies[0].shot_ids).toEqual(["s1", "h0003"]);
    expect(out.corrections_skipped).toEqual(["cor_5"]);
  });

  it("re-homes shots when rally bounds change; swap_player_ids flips hitters and winners", () => {
    seq = 0;
    const out = applyCorrections(
      base,
      [
        corr({ op: "set_rally_bounds", payload: { rally_id: "r1", end_frame: 60 } }),
        corr({ op: "add_rally", payload: { start_frame: 3, end_frame: 27 } }),
        corr({ op: "swap_player_ids", payload: { from_frame: 200 } }),
        corr({ op: "confirm_rally", payload: { rally_id: "r2" } }),
      ],
      FPS,
    );
    expect(out.rallies.map((r) => r.rally_id)).toEqual(["hr0002", "r1", "r2"]);
    expect(out.rallies[1]).toMatchObject({ end_ms: 2000, end_reason: "manual", source: "human" });
    expect(out.shots.find((s) => s.shot_id === "s2")!.rally_id).toBeNull();
    expect(out.shots.find((s) => s.shot_id === "s3")!.hitter).toBe("P1");
    expect(out.rallies[2]).toMatchObject({ winner: "P2", status: "ok", source: "human" });
  });

  // Parity with the backend: raw example + example corrections == backend's effective example.
  const ex = resolve(__dirname, "../../../contracts/examples");
  it.skipIf(!existsSync(resolve(ex, "events_effective.json")))("matches backend contracts/examples/events_effective.json", () => {
    const read = (f: string) => JSON.parse(readFileSync(resolve(ex, f), "utf8"));
    const meta = read("video_meta.json");
    const raw = existsSync(resolve(ex, "strokes.json")) ? mergeStrokes(read("events.json"), read("strokes.json")) : read("events.json");
    const out = applyCorrections(raw, read("corrections.json"), meta.fps);
    const expected = read("events_effective.json");
    expect(out.shots).toEqual(expected.shots);
    expect(out.rallies).toEqual(expected.rallies);
  });
});

describe("export", () => {
  it("writes the backend CSV columns, one row per shot, empty cells for unknowns", () => {
    seq = 0;
    const csv = exportCsv(applyCorrections(base, [corr({ op: "add_shot", payload: { frame: 600, hitter: null } })], FPS));
    const lines = csv.trim().split("\r\n");
    expect(lines).toHaveLength(5);
    expect(lines[0]).toBe(CSV_HEADER.join(","));
    expect(lines[4]).toBe("h0001,,600,20000,00:20.000,,,1.000,ok,human,,,,,,,,,,,,");
    // shots inside a rally carry its number, their position in it and the rally size
    const first = lines[1].split(",");
    expect(first.slice(-4, -1)).toEqual(["1", "1", String(base.rallies[0].shot_ids.length)]);
  });
  it("adds stroke-type columns", () => {
    seq = 0;
    const csv = exportCsv(applyCorrections(base, [corr({ op: "set_shot_type", payload: { shot_id: "s1", shot_type: "smash" } })], FPS));
    expect(csv.split("\r\n")[1].includes(",smash,1.000,ok,human,")).toBe(true);
  });
});

describe("stroke types", () => {
  const strokes = {
    video_id: "v",
    classifier: "rules_v1",
    shots: [
      { shot_id: "s1", frame: 45, stroke: { shot_type: "long_serve" as const, confidence: 0.8, status: "ok" as const, source: "model" as const } },
      { shot_id: "s2", frame: 75, stroke: { shot_type: "clear" as const, confidence: 0.4, status: "needs_review" as const, source: "model" as const } },
      { shot_id: "s3", frame: 270, stroke: { shot_type: null, confidence: 0.1, status: "undeterminable" as const, source: "model" as const } },
    ],
  };
  it("merges predictions without touching raw events", () => {
    const snap = JSON.stringify(base);
    const m = mergeStrokes(base, strokes);
    expect(JSON.stringify(base)).toBe(snap);
    expect(m.shots.map((s) => s.shot_type)).toEqual(["long_serve", "clear", null]);
    expect(mergeStrokes(base, null)).toBe(base);
  });
  it("set_shot_type relabels only the stroke; unknown shots are skipped", () => {
    seq = 0;
    const out = applyCorrections(
      mergeStrokes(base, strokes),
      [corr({ op: "set_shot_type", payload: { shot_id: "s2", shot_type: "drop" } }), corr({ op: "set_shot_type", payload: { shot_id: "zz", shot_type: "drop" } })],
      FPS,
    );
    const s2 = out.shots.find((s) => s.shot_id === "s2")!;
    expect(s2).toMatchObject({ shot_type: "drop", status: "needs_review", source: "model" });
    expect(s2.stroke).toMatchObject({ shot_type: "drop", source: "human", status: "ok" });
    expect(out.corrections_skipped).toEqual(["cor_2"]);
  });
  it("counts per-player distributions and pending reviews", () => {
    const [p1, p2] = strokeStats(mergeStrokes(base, strokes).shots, ["P1", "P2"]);
    expect(p1).toMatchObject({ total: 2, typed: 2, review: 1 });
    expect(p1.counts.map((c) => c.type)).toEqual(["long_serve", "clear"]);
    expect(p2).toMatchObject({ total: 1, typed: 0, counts: [] });
  });
  it("mock data carries stroke predictions", () => {
    const m = generateMock("demo", { durationMs: 20_000 });
    expect(m.events.shots.some((s) => s.shot_type)).toBe(true);
  });
});

describe("overlay, heatmap, jobs", () => {
  const mock = generateMock("demo", { durationMs: 20_000 });
  it("finds frames by time", () => {
    const f = mock.overlay.frames;
    expect(frameIndexAt(f, -5)).toBe(0);
    expect(frameIndexAt(f, 1e9)).toBe(f.length - 1);
    const i = frameIndexAt(f, 1000);
    expect(f[i].t_ms).toBeLessThanOrEqual(1000);
    expect(f[i + 1].t_ms).toBeGreaterThan(1000);
  });
  it("counts player samples on court", () => {
    const h = computeHeat(mock.overlay.frames, "P1", 0, 20_000);
    expect(h.samples).toBeGreaterThan(500);
    expect(h.counts.flat().reduce((a, b) => a + b, 0)).toBe(h.samples);
    expect(h.distanceM).toBeGreaterThan(0);
    // P1 is the near player: y = 0 is the near baseline, so most samples are in the near half.
    const nearRows = h.counts.slice(0, Math.floor(h.rows / 2)).flat().reduce((a, b) => a + b, 0);
    expect(nearRows / h.samples).toBeGreaterThan(0.9);
  });
  it("mock is deterministic and contains review items", () => {
    expect(generateMock("demo", { durationMs: 20_000 }).events).toEqual(mock.events);
    expect(mock.events.shots.some((s) => s.status === "needs_review")).toBe(true);
  });
  it("job progress ignores skipped stages", () => {
    expect(jobProgress([{ state: "skipped", progress: 0 }, { state: "succeeded", progress: 1 }, { state: "running", progress: 0.5 }])).toBeCloseTo(0.75);
  });
});

describe("doubles", () => {
  const mock = generateMock("doubles", { durationMs: 30_000, doubles: true });

  it("mock has four players in two teams and a doubles boundary", () => {
    expect(mock.overlay.match_format).toBe("doubles");
    const f = mock.overlay.frames[10];
    expect(f.players.map((p) => p.player_id).sort()).toEqual(["P1", "P2", "P3", "P4"]);
    expect(f.players.find((p) => p.player_id === "P3")?.team).toBe("A");
    expect(mock.overlay.court[0].boundary).toHaveLength(4);
    // hits alternate between teams
    for (const r of mock.events.rallies) {
      const teams = mock.events.shots.filter((s) => r.shot_ids.includes(s.shot_id) && s.source === "model").map((s) => (s.hitter === "P1" || s.hitter === "P3" ? "A" : "B"));
      expect(teams.length).toBeGreaterThan(0);
    }
    expect(new Set(mock.events.shots.map((s) => s.hitter))).toEqual(new Set(["P1", "P2", "P3", "P4"]));
  });

  it("team heatmap counts both partners and sums their distance", () => {
    const one = computeHeat(mock.overlay.frames, "P1", 0, 30_000);
    const other = computeHeat(mock.overlay.frames, "P3", 0, 30_000);
    const team = computeHeat(mock.overlay.frames, ["P1", "P3"], 0, 30_000);
    expect(team.samples).toBe(one.samples + other.samples);
    expect(team.distanceM).toBeCloseTo(one.distanceM + other.distanceM, 6);
  });

  it("swap within a team keeps the team winner; across teams swaps it", () => {
    const ev: EventsResult = {
      ...base,
      match_format: "doubles",
      rallies: base.rallies.map((r) => ({ ...r, winner: "A" })),
      shots: base.shots.map((s, i) => ({ ...s, hitter: (["P1", "P3", "P4"] as const)[i] })),
    };
    const within = applyCorrections(ev, [corr({ op: "swap_player_ids", payload: { from_frame: 0, a: "P1", b: "P3" } })], FPS);
    expect(within.shots.map((s) => s.hitter)).toEqual(["P3", "P1", "P4"]);
    expect(within.rallies.map((r) => r.winner)).toEqual(["A", "A"]);
    const across = applyCorrections(ev, [corr({ op: "swap_player_ids", payload: { from_frame: 0, a: "P3", b: "P4" } })], FPS);
    expect(across.shots.map((s) => s.hitter)).toEqual(["P1", "P4", "P3"]);
    expect(across.rallies.map((r) => r.winner)).toEqual(["B", "B"]);
  });
});

describe("switching the analysed court", () => {
  it("finds the other court under a click and keeps ignoring the old one", async () => {
    const { otherCourtAt, courtsAfterSwitch, pointInPolygon } = await import("./overlay");
    const m = generateMock("vid_multi", { durationMs: 5_000, width: 1280, height: 720 });
    const cal = m.overlay.court[0];
    expect(cal.other_courts?.length).toBe(1);
    const other = cal.other_courts![0];
    const cx = other.reduce((s, p) => s + p[0], 0) / 4;
    const cy = other.reduce((s, p) => s + p[1], 0) / 4;
    expect(otherCourtAt(cal, [cx, cy])).toBe(other);
    expect(otherCourtAt(cal, [640, 500])).toBeNull(); // our own court
    const after = courtsAfterSwitch(cal, other);
    expect(after).toEqual([cal.image_corners]);
    expect(pointInPolygon(cal.image_corners!, [640, 500])).toBe(true);
  });
});

describe("retypeDetail", () => {
  // expected texts produced by rallyiq.strokes.detail.retype on the same input
  const d = {
    family: "drop", subtype: null, source: "attrs_v1", description_zh: "x", role: null, intent: null,
    hand: { value: "backhand", p: 0.7, source: "model" as const },
    direction: { value: "straight", p: 0.9, source: "model" as const },
    contact: { value: "high", p: 0.8, source: "model" as const },
    pace: { value: "slow", p: null, source: "measured" as const },
  };
  it("matches the backend wording", () => {
    expect(retypeDetail(d, "smash")?.description_zh).toBe("反手直线杀球，击球点较高，属于主动进攻，意图是直接施压，逼对手被动或失误。");
    expect(retypeDetail(d, "cross_net")?.description_zh).toBe("反手勾对角，击球点较高，属于相持，意图是控制网前，逼对手起球，为下一拍进攻创造机会。");
    expect(retypeDetail(d, "long_serve")).toMatchObject({ family: "serve", subtype: "long_serve", hand: null, direction: null, pace: null });
    expect(retypeDetail(d, "drop")?.pace).toEqual(d.pace);
    expect(retypeDetail(null, "smash")).toBeNull();
  });
});

describe("upload file check", () => {
  it("accepts every extension the backend accepts and nothing else", () => {
    for (const name of ["a.mp4", "b.MOV", "c.m4v", "d.mkv", "e.avi", "f.webm"]) expect(isVideoFile({ name, type: "" })).toBe(true);
    expect(isVideoFile({ name: "clip", type: "video/mp4" })).toBe(true);
    expect(isVideoFile({ name: "x.3gp", type: "video/3gpp" })).toBe(false);
    expect(isVideoFile({ name: "notes.txt", type: "text/plain" })).toBe(false);
  });
});
