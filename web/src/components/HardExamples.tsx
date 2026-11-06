import { useEffect, useState } from "react";
import { api } from "../lib/api";
import type { ConditionGroup, HardExampleSummary } from "../lib/types";

const CONDITION_ZH: Record<string, string> = {
  camera_view: "机位", camera_height: "机位高度", fps: "帧率", resolution: "分辨率", light: "光照", match_format: "单/双打",
};
const VALUE_ZH: Record<string, string> = {
  end_center: "后场正中", end_oblique: "后场偏斜", side: "侧面", high: "高位", medium: "中等高度", low: "低位",
  unknown: "未知", dark: "偏暗", bright: "过曝", normal: "正常", singles: "单打", doubles: "双打",
};
const MODULE_ZH: Record<string, string> = { hits: "击球", rallies: "回合", players: "球员", strokes: "类型" };

export const groupLabel = (g: ConditionGroup) => `${CONDITION_ZH[g.condition] ?? g.condition}：${VALUE_ZH[g.value] ?? g.value}`;

/** Corrections marked "记为难例", grouped by the conditions they happened under (README: 每周挑出最弱的场景). */
export function HardExamples() {
  const [s, setS] = useState<HardExampleSummary | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    api.getHardExampleSummary().then(setS, (e: Error) => setError(e.message));
  }, []);
  const url = api.hardExamplesUrl();
  return (
    <section className="card">
      <h2>难例与薄弱场景</h2>
      {error ? (
        <p className="muted">无法加载：{error}</p>
      ) : !s ? (
        <p className="muted">加载中…</p>
      ) : s.total === 0 ? (
        <p className="muted">还没有难例。在工作台修正时勾选“记为难例”（默认勾选），修正会自动收进评估集，并按机位、帧率、光照分组统计。</p>
      ) : (
        <>
          <p className="small">
            {s.videos} 个视频中共 {s.total} 条难例，其中 {s.errors} 条是模型错误。
            {url && (
              <>
                {" "}
                <a href={url}>下载 JSONL</a>
              </>
            )}
          </p>
          {s.weakest.length > 0 && (
            <>
              <h3>击球错误率最高的条件</h3>
              <table>
                <thead>
                  <tr>
                    <th>条件</th>
                    <th>视频</th>
                    <th>模型击球</th>
                    <th>击球错误率</th>
                    <th>各模块错误</th>
                  </tr>
                </thead>
                <tbody>
                  {s.weakest.map((g) => (
                    <tr key={`${g.condition}:${g.value}`}>
                      <td>{groupLabel(g)}</td>
                      <td className="mono">{g.videos}</td>
                      <td className="mono">{g.model_shots}</td>
                      <td className="mono">{g.hit_error_rate == null ? "—" : `${(g.hit_error_rate * 100).toFixed(1)}%`}</td>
                      <td className="small muted">
                        {Object.entries(g.errors)
                          .map(([m, n]) => `${MODULE_ZH[m] ?? m} ${n}`)
                          .join(" · ") || "—"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </>
          )}
        </>
      )}
    </section>
  );
}
