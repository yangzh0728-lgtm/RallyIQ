import { useEffect, useState } from "react";
import { api } from "../lib/api";
import { capabilityHeadline, sortChecks } from "../lib/quality";
import type { VideoQuality } from "../lib/types";
import { StatusPill } from "./CorrectionPanel";

/** "这段视频能信多少": camera view, picture, visibility and per-output reliability. */
export function QualityPanel({ videoId, refreshKey }: { videoId: string; refreshKey?: unknown }) {
  const [q, setQ] = useState<VideoQuality | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    let live = true;
    api.getQuality(videoId).then(
      (v) => live && (setQ(v), setError(null)),
      (e: Error) => live && setError(e.message),
    );
    return () => {
      live = false;
    };
  }, [videoId, refreshKey]);

  if (error) return <div className="panel small muted">视频评估暂不可用：{error}</div>;
  if (!q) return <div className="panel small muted">评估视频条件…</div>;
  const wrongCourt = q.checks.find((c) => c.key === "court_choice" && c.status !== "ok");
  return (
    <div className="panel small quality">
      <div className="panel-head">
        <h3>视频条件</h3>
        <span className="muted">{capabilityHeadline(q.capabilities)}</span>
      </div>
      {wrongCourt && <div className="notice warn">{wrongCourt.note}</div>}
      <div className="quality-summary">{q.summary}</div>
      <div className="quality-caps">
        {q.capabilities.map((c) => (
          <div key={c.capability} className="kv" title={c.reasons.join("\n") || undefined}>
            <span>{c.label}</span>
            <span className="v">
              <StatusPill status={c.status} />
            </span>
          </div>
        ))}
      </div>
      <details>
        <summary className="muted">逐项检查（{q.checks.length}）</summary>
        <div className="quality-checks">
          {sortChecks(q.checks).map((c) => (
            <div key={c.key} className="quality-check">
              <div className="kv">
                <span className="k">{c.label}</span>
                <span className="v">
                  <span className="mono">{c.value ?? "—"}</span>
                  <StatusPill status={c.status} />
                </span>
              </div>
              {c.note && <div className="muted note">{c.note}</div>}
            </div>
          ))}
        </div>
      </details>
    </div>
  );
}
