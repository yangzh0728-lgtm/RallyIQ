import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { generateMock } from "./mock";
import { capabilityHeadline, mockQuality, sortChecks } from "./quality";
import type { VideoQuality } from "./types";

describe("quality", () => {
  it("puts the weakest checks first and counts capabilities", () => {
    const q = mockQuality(generateMock("demo", { durationMs: 20_000 }).overlay);
    expect(sortChecks(q.checks)[0].status).toBe("needs_review");
    expect(capabilityHeadline(q.capabilities)).toBe("2 项可用 · 3 项待复核");
  });
  it("matches the backend example contract", () => {
    const p = resolve(__dirname, "../../../contracts/examples/quality.json");
    if (!existsSync(p)) return;
    const q = JSON.parse(readFileSync(p, "utf8")) as VideoQuality;
    expect(q.capabilities.map((c) => c.capability)).toEqual(["court_metrics", "players", "shuttle", "rallies", "hits"]);
    expect(q.checks.every((c) => typeof c.key === "string" && ["ok", "needs_review", "undeterminable"].includes(c.status))).toBe(true);
  });
});
