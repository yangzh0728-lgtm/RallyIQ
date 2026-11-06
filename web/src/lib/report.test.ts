import { describe, expect, it } from "vitest";
import { applyCorrections } from "./corrections";
import { generateMock } from "./mock";
import { fmtValue, indexStats, isMetric, mockCoach, mockStats, reportMarkdown } from "./report";

describe("report (mock)", () => {
  const a = generateMock("demo");
  const stats = mockStats(a.events, a.overlay);
  const idx = indexStats(stats);

  it("computes rally and player stats without inventing values", () => {
    const rallies = idx.get("match.rallies")!;
    expect(isMetric(rallies) && rallies.value).toBe(a.events.rallies.length);
    const p1 = idx.get("P1.shots")!;
    const p2 = idx.get("P2.shots")!;
    expect(isMetric(p1) && isMetric(p2) && p1.value! + p2.value!).toBe(a.events.shots.filter((s) => s.hitter).length);
    const d = idx.get("P1.move.distance_m")!;
    expect(isMetric(d) && d.value! > 0).toBe(true);
    const types = idx.get("P1.shot_types")!;
    expect(types.status).toBe("undeterminable");
  });

  it("fingerprint follows corrections", () => {
    const r = a.events.rallies[0];
    const ev = applyCorrections(a.events, [
      { op: "set_rally_winner", payload: { rally_id: r.rally_id, winner: r.winner === "P1" ? "P2" : "P1" }, correction_id: "c1", video_id: "demo", created_at: "", seq: 1, note: null, hard_example: true },
    ], a.overlay.video.fps);
    expect(mockStats(ev, a.overlay).fingerprint).not.toBe(stats.fingerprint);
  });

  it("coach cites existing evidence and renders markdown", () => {
    const c = mockCoach(stats);
    for (const i of c.insights) for (const k of i.evidence) expect(idx.has(k)).toBe(true);
    const md = reportMarkdown("赛后报告", stats, c, false);
    expect(md).toContain("AI 教练点评");
    expect(md).toContain("回合数");
  });

  it("formats values", () => {
    expect(fmtValue({ value: 66.66, unit: "%" })).toBe("66.7%");
    expect(fmtValue({ value: 12, unit: "m" })).toBe("12 m");
    expect(fmtValue({ value: null, unit: "m" })).toBe("—");
  });
});
