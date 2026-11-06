import type { CapabilityReliability, OverlayData, QualityCheck, Status, VideoQuality } from "./types";

const ORDER: Record<Status, number> = { ok: 0, needs_review: 1, undeterminable: 2 };

/** Worst status first, then by label: what the user should look at first. */
export function sortChecks(checks: QualityCheck[]): QualityCheck[] {
  return [...checks].sort((a, b) => ORDER[b.status] - ORDER[a.status] || a.label.localeCompare(b.label));
}

/** Short headline for the panel: "3 项可用 · 2 项待复核". */
export function capabilityHeadline(caps: CapabilityReliability[]): string {
  const n = (s: Status) => caps.filter((c) => c.status === s).length;
  const parts = [`${n("ok")} 项可用`];
  if (n("needs_review")) parts.push(`${n("needs_review")} 项待复核`);
  if (n("undeterminable")) parts.push(`${n("undeterminable")} 项不可判定`);
  return parts.join(" · ");
}

/** Synthetic assessment for ?mock mode (the real one is computed by rallyiq/ingest/quality.py). */
export function mockQuality(overlay: OverlayData): VideoQuality {
  const v = overlay.video;
  const short = Math.min(v.width, v.height);
  const check = (key: string, label: string, value: string, status: Status, note: string | null = null): QualityCheck => ({ key, label, value, status, note });
  const checks: QualityCheck[] = [
    check("fps", "帧率", `${Math.round(v.fps)} fps`, v.fps < 24.5 ? "needs_review" : "ok", v.fps < 49 ? "30 fps 左右：快速杀球会有运动模糊和丢帧，击球按区域级精度给出" : null),
    check("resolution", "分辨率", `${v.width}×${v.height}`, short < 540 ? "needs_review" : "ok"),
    check("calibration", "球场标定", "100% 时长已标定", "ok"),
    check("camera_view", "机位", "后场正中", "ok"),
    check("camera_height", "机位高度", "中等高度", "ok"),
    check("far_player_rate", "远端球员可见", "回合内 86%", "ok"),
    check("shuttle_rate", "羽毛球检出", "回合内 58%", "needs_review", "球的检出率低：回合切分和击球时刻会漏判，补点和缺口已单独标记"),
  ];
  const cap = (capability: CapabilityReliability["capability"], label: string, status: Status, reasons: string[] = []): CapabilityReliability => ({ capability, label, status, reasons });
  return {
    schema_version: "0.2.0",
    video_id: v.video_id,
    fingerprint: "mock",
    camera_view: "end_center",
    camera_height: "medium",
    checks,
    capabilities: [
      cap("court_metrics", "球场坐标（米、m/s、热力图）", "ok"),
      cap("players", "球员跑位与身份", "ok"),
      cap("shuttle", "羽毛球轨迹", "needs_review", ["羽毛球检出：回合内 58%"]),
      cap("rallies", "回合切分", "needs_review", ["羽毛球检出：回合内 58%"]),
      cap("hits", "击球与击球者", "needs_review", ["羽毛球检出：回合内 58%", "击球检测尚未达到 F1 ≥ 0.90 的去复核门槛"]),
    ],
    summary: "后场正中机位（中等高度），30 fps；需要注意：羽毛球检出；可直接使用：球场坐标（米、m/s、热力图）、球员跑位与身份。（演示数据）",
  };
}
