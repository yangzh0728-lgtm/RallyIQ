import { useEffect, useState } from "react";
import { api } from "../lib/api";
import type { HardExampleSummary } from "../lib/types";
import { Empty, Icon } from "./ui";

const CONDITION_ZH: Record<string, string> = {
  camera_view: "机位", camera_height: "机位高度", fps: "帧率", resolution: "分辨率", light: "光照", match_format: "单/双打",
};
const VALUE_ZH: Record<string, string> = {
  end_center: "后场正中", end_oblique: "后场偏斜", side: "侧面", high: "高位", medium: "中等高度", low: "低位",
  unknown: "未知", dark: "偏暗", bright: "过曝", normal: "正常", singles: "单打", doubles: "双打",
};
const MODULE_ZH: Record<string, string> = { hits: "击球", rallies: "回合", players: "球员", strokes: "类型" };


/** Corrections marked "记为难例", grouped by the conditions they happened under (README: 每周挑出最弱的场景). */
export function HardExamples() {
  const [s, setS] = useState<HardExampleSummary | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    api.getHardExampleSummary().then(setS, (e: Error) => setError(e.message));
  }, []);
  const url = api.hardExamplesUrl();
  const hasData = !error && !!s && s.total > 0;
  // Bars compare the listed conditions with each other; the percentage beside them is the real rate.
  return (
    <section className="card hard-card">
      <div className="card-head">
        <h2>难例与薄弱场景</h2>
        <span className="sub">人工修正会积累成评估集，按拍摄条件找出模型最弱的场景</span>
        <span className="spacer" />
        {hasData && url && (
          <a className="btn small" href={url} title="下载全部难例（JSONL）">
            <Icon name="download" />
            下载 JSONL
          </a>
        )}
      </div>
      {error ? (
        <Empty icon="alert" title="无法加载难例统计">{error}</Empty>
      ) : !s ? (
        <p className="muted small">加载中…</p>
      ) : s.total === 0 ? (
        <Empty icon="target" title="还没有难例">
          在工作台修正时勾选“记为难例”（默认勾选），修正会自动收进评估集，并按机位、帧率、光照分组统计。
        </Empty>
      ) : (
        <>
          <div className="hard-stats">
            <div className="metric">
              <div className="label">难例总数</div>
              <div className="value mono">
                {s.total}
                <small>条</small>
              </div>
              <div className="note">修正时勾选“记为难例”的记录</div>
            </div>
            <div className="metric">
              <div className="label">其中模型错误</div>
              <div className="value mono">
                {s.errors}
                <small>条</small>
              </div>
              <div className="note">
                占全部难例的 <span className="mono">{Math.round((s.errors / s.total) * 100)}%</span>
              </div>
            </div>
            <div className="metric">
              <div className="label">涉及视频</div>
              <div className="value mono">
                {s.videos}
                <small>个</small>
              </div>
              <div className="note">按机位、帧率、光照等条件分组</div>
            </div>
          </div>
          {s.weakest.length > 0 && (
            <>
              <h3>击球错误率最高的条件</h3>
              <div className="table-scroll">
                <table className="hard-table">
                  <thead>
                    <tr>
                      <th>条件</th>
                      <th className="num">视频</th>
                      <th className="num">模型击球</th>
                      <th>击球错误率</th>
                      <th>各模块错误</th>
                    </tr>
                  </thead>
                  <tbody>
                    {s.weakest.map((g) => {
                      const mods = Object.entries(g.errors);
                      return (
                        <tr key={`${g.condition}:${g.value}`}>
                          <td>
                            <span className="muted">{CONDITION_ZH[g.condition] ?? g.condition}</span>
                            <b className="cond-value">{VALUE_ZH[g.value] ?? g.value}</b>
                          </td>
                          <td className="num mono">{g.videos}</td>
                          <td className="num mono">{g.model_shots}</td>
                          <td>
                            {g.hit_error_rate == null ? (
                              <span className="muted">—</span>
                            ) : (
                              <div className="pbar">
                                <span className="rate mono">{(g.hit_error_rate * 100).toFixed(1)}%</span>
                                <div className="bar">
                                  <i style={{ width: `${Math.min(1, g.hit_error_rate) * 100}%` }} />
                                </div>
                              </div>
                            )}
                          </td>
                          <td>
                            {mods.length ? (
                              <div className="row" style={{ gap: 4 }}>
                                {mods.map(([m, n]) => (
                                  <span key={m} className="chip">
                                    {MODULE_ZH[m] ?? m} <span className="mono">{n}</span>
                                  </span>
                                ))}
                              </div>
                            ) : (
                              <span className="muted">—</span>
                            )}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </>
          )}
        </>
      )}
    </section>
  );
}
