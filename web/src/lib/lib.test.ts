import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { applyCorrections } from "./corrections";
import { applyH, CORNERS_M, homographyFrom4, invertH } from "./court";
import { CSV_HEADER, exportCsv } from "./export";
import { computeHeat } from "./heatmap";
import { generateMock } from "./mock";
import { frameIndexAt } from "./overlay";
import { jobProgress } from "./players";
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
    const out = applyCorrections(read("events.json"), read("corrections.json"), meta.fps);
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
    expect(lines[4]).toBe("h0001,,600,20000,00:20.000,,1.000,ok,human,,,,");
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
