import { describe, expect, it } from "vitest";
import { applyCorrections } from "./corrections";
import { compareRows, leadSeries, rallyLog } from "./flow";
import { generateMock } from "./mock";
import { clock, fmtValue, indexStats, isMetric, mockCoach, mockStats, reportMarkdown, styleMarkdown } from "./report";
import type { EventsResult, PlayerStyle } from "./types";

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

describe("playing style", () => {
  const style: PlayerStyle = {
    subject: "P1", players: ["P1"], kind: "singles", label: "强攻压制型",
    primary: { code: "power_attack", name: "强攻压制型", name_en: "Power Attacking", score: 82.4, coverage: 1, reasons: ["杀球占比 22%（偏高）"], representatives: ["安赛龙 Viktor Axelsen"] },
    secondary: null,
    scores: [
      { code: "power_attack", name: "强攻压制型", name_en: "Power Attacking", score: 82.4, coverage: 1, reasons: [], representatives: [] },
      { code: "tempo", name: "节奏变化型", name_en: "Tempo Variation", score: null, coverage: 0.2, reasons: [], representatives: [] },
    ],
    phases: [
      { phase: "attack", shots: 10, counts: { smash: 8, clear: 2 }, recommended_pct: 80 },
      { phase: "defense", shots: 0, counts: {}, recommended_pct: null },
    ],
    tactics: [{ code: "net_to_attack", name: "网前逼起球", uses: 4, decided: 3, won: 2, success_pct: 66.7, forced_pct: 50, examples_ms: [61000] }],
    habits: [{ after: "smash", reply: "block", n: 6, of: 8, pct: 75 }],
    evidence: ["P1.style.smash_pct"], advice: ["多用网前逼起球"], n_shots: 60, status: "needs_review", confidence: "medium", note: null,
  };

  it("renders the style section", () => {
    const stats = mockStats(generateMock("demo").events, generateMock("demo").overlay);
    expect(styleMarkdown(stats)).toEqual([]);
    const md = styleMarkdown({ ...stats, styles: [style] }).join("\n");
    expect(md).toContain("### P1：强攻压制型");
    expect(md).toContain("吻合度 82");
    expect(md).not.toContain("节奏变化型 "); // unscored prototypes are left out
    expect(md).toContain("主动状态 10 拍：杀球 8，高远球 2；符合推荐打法 80%");
    expect(md).toContain("| 网前逼起球 | 4 | 3 | 2 | 67% | 50% |");
    expect(md).toContain("接杀球时 75% 回挡网（6/8）");
    expect(reportMarkdown("t", { ...stats, styles: [style] }, null, false)).toContain("## 打法风格");
    expect(clock(61000)).toBe("01:01");
    expect(clock(3661000)).toBe("1:01:01");
  });
});

describe("match flow", () => {
  const ev = (winners: (string | null)[]): EventsResult => ({
    schema_version: "0.1.0",
    video_id: "v",
    match_format: "singles",
    shots: winners.flatMap((_, i) =>
      ["P1", "P2"].map((h, k) => ({
        shot_id: `s${i}_${k}`, rally_id: `r${i}`, frame: i * 300 + 15 + k * 24, t_ms: i * 10000 + 500 + k * 800, hitter: h,
        confidence: 0.9, status: "needs_review", source: "model", shuttle_px: null, hitter_court: null, shot_type: k ? "smash" : null,
      })),
    ),
    rallies: winners.map((w, i) => ({
      rally_id: `r${i}`, start_frame: i * 300, end_frame: i * 300 + 60, start_ms: i * 10000, end_ms: i * 10000 + 2000,
      shot_ids: [`s${i}_0`, `s${i}_1`], end_reason: "unknown", winner: w, confidence: 0.9, status: "needs_review", source: "model",
    })),
  }) as unknown as EventsResult;

  it("rebuilds games with the backend rules (21, win by 2)", () => {
    const { rallies, games } = rallyLog(ev([...Array(21).fill("P2"), "P1", null]), (t) => (t === "smash" ? "杀球" : t));
    expect(games[0]).toEqual({ game_no: 1, score: { P1: 0, P2: 21 }, winner: "P2", rallies: 21 });
    expect(rallies[21].score).toEqual({ P1: 1, P2: 0 });
    expect(rallies[22].score).toBeNull();
    expect(rallies[0].last_shot_type).toBe("杀球");
    expect(rallies[0].mean_interval_s).toBe(0.8);
    expect(leadSeries(rallies, ["P1", "P2"]).map((x) => x.lead).slice(0, 2)).toEqual([-1, -2]);
  });

  it("demo stats carry a rally log and a comparison", () => {
    const a = generateMock("demo");
    const stats = mockStats(a.events, a.overlay);
    expect(stats.rallies).toHaveLength(a.events.rallies.length);
    expect(compareRows(stats.players).every((r) => r.values.length === 2)).toBe(true);
  });
});
