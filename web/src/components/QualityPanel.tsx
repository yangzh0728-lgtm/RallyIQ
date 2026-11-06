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
      <h3>
        视频条件 <span className="muted">· {capabilityHeadline(q.capabilities)}</span>
      </h3>
      {wrongCourt && <div className="notice warn">{wrongCourt.note}</div>}
      <div className="quality-summary">{q.summary}</div>
      <div className="quality-caps">
        {q.capabilities.map((c) => (
          <div key={c.capability} className="row" title={c.reasons.join("\n") || undefined}>
            <span>{c.label}</span>
            <span className="spacer" />
            <StatusPill status={c.status} />
          </div>
        ))}
      </div>
      <details>
        <summary className="muted">逐项检查（{q.checks.length}）</summary>
        <div className="quality-checks">
          {sortChecks(q.checks).map((c) => (
            <div key={c.key} className="quality-check">
              <div className="row">
                <span className="muted">{c.label}</span>
                <span className="spacer" />
                <span className="mono">{c.value ?? "—"}</span>
                <StatusPill status={c.status} />
              </div>
              {c.note && <div className="muted note">{c.note}</div>}
            </div>
          ))}
        </div>
      </details>
    </div>
  );
}
