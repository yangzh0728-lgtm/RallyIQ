import { useCallback, useEffect } from "react";
import { api } from "../lib/api";
import { jobProgress } from "../lib/players";
import { usePoll } from "../lib/usePoll";
import type { Job } from "../lib/types";

const JOB_LABEL: Record<string, string> = {
  queued: "排队中",
  running: "分析中",
  succeeded: "完成",
  failed: "失败",
  cancelled: "已取消",
  pending: "等待",
  skipped: "跳过",
};
const STAGE_LABEL: Record<string, string> = {
  probe: "视频质量检查",
  calibration: "球场标定",
  players: "球员与骨架",
  shuttle: "羽毛球轨迹",
  events: "回合与击球",
  strokes: "击球类型",
  feet: "脚跟与脚尖",
  overlay: "生成叠加层",
};
export const jobFinished = (j: Job) => j.state === "succeeded" || j.state === "failed" || j.state === "cancelled";

export function useJob(jobId: string | null) {
  const fetchJob = useCallback(() => api.getJob(jobId!), [jobId]);
  return usePoll(jobId ? fetchJob : null, 1000, jobFinished);
}

export function JobBadge({ job: initial, onDone }: { job: Job | null; onDone?: () => void }) {
  const live = useJob(initial && !jobFinished(initial) ? initial.job_id : null);
  const job = live.value ?? initial;
  // A job that finishes while the list is open: reload so duration / size / links appear.
  const finishedLive = !!live.value && jobFinished(live.value);
  useEffect(() => {
    if (finishedLive) onDone?.();
  }, [finishedLive, onDone]);
  if (!job) return <span className="muted">未分析</span>;
  const p = job.state === "succeeded" ? 1 : jobProgress(job.stages);
  const cls = job.state === "succeeded" ? "ok" : job.state === "failed" ? "bad" : "";
  return (
    <div style={{ minWidth: 160 }} title={job.error ?? undefined}>
      <div className="row small">
        <span>{JOB_LABEL[job.state] ?? job.state}</span>
        <span className="spacer" />
        <span className="muted mono">{Math.round(p * 100)}%</span>
      </div>
      <div className={`bar ${cls}`}>
        <i style={{ width: `${p * 100}%` }} />
      </div>
    </div>
  );
}

export function JobDetail({ job, title, onRetry }: { job: Job; title?: string; onRetry?: () => void }) {
  const p = jobProgress(job.stages);
  return (
    <div className="card" style={{ maxWidth: 520, margin: "40px auto" }}>
      <h2>
        {title ?? "分析"}
        {JOB_LABEL[job.state]} · <span className="mono">{Math.round(p * 100)}%</span>
      </h2>
      <div style={{ display: "grid", gap: 10 }}>
        {job.stages.map((s) => (
          <div key={s.name}>
            <div className="row small">
              <span>{STAGE_LABEL[s.name] ?? s.name}</span>
              <span className="spacer" />
              <span className={s.error ? "error" : "muted"}>{s.error ?? JOB_LABEL[s.state] ?? s.state}</span>
            </div>
            <div className={`bar ${s.state === "succeeded" ? "ok" : s.state === "failed" ? "bad" : ""}`}>
              <i style={{ width: `${(s.state === "succeeded" ? 1 : s.progress) * 100}%` }} />
            </div>
          </div>
        ))}
      </div>
      {job.error && <p className="error">{job.error}</p>}
      {onRetry && (job.state === "failed" || job.state === "cancelled") && (
        <div className="row" style={{ marginTop: 12 }}>
          <button className="primary" onClick={onRetry}>
            重新分析
          </button>
          <span className="muted small">从头重新跑一遍全部阶段；已做的人工修正会保留。</span>
        </div>
      )}
    </div>
  );
}
