import { useCallback, useEffect } from "react";
import { api } from "../lib/api";
import { jobProgress } from "../lib/players";
import { usePoll } from "../lib/usePoll";
import type { Job } from "../lib/types";
import { Icon } from "./ui";

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
  overlay: "生成叠加层",
};
/** Pill variant per job state: a dot plus the label, never colour alone. */
const JOB_PILL: Record<string, string> = { succeeded: "ok", running: "running", failed: "bad" };
/** Heading wording per job state ("分析" + …). */
const JOB_PHRASE: Record<string, string> = { queued: "排队中", running: "进行中", succeeded: "完成", failed: "失败", cancelled: "已取消" };

export const jobFinished = (j: Job) => j.state === "succeeded" || j.state === "failed" || j.state === "cancelled";

export function useJob(jobId: string | null) {
  const fetchJob = useCallback(() => api.getJob(jobId!), [jobId]);
  return usePoll(jobId ? fetchJob : null, 1000, jobFinished);
}

const stageLabel = (name: string) => STAGE_LABEL[name] ?? name;

export function JobBadge({ job: initial, onDone }: { job: Job | null; onDone?: () => void }) {
  const live = useJob(initial && !jobFinished(initial) ? initial.job_id : null);
  const job = live.value ?? initial;
  // A job that finishes while the list is open: reload so duration / size / links appear.
  const finishedLive = !!live.value && jobFinished(live.value);
  useEffect(() => {
    if (finishedLive) onDone?.();
  }, [finishedLive, onDone]);
  if (!job)
    return (
      <div className="job-badge">
        <div className="job-line">
          <span className="pill outline muted">未分析</span>
        </div>
      </div>
    );
  const p = job.state === "succeeded" ? 1 : jobProgress(job.stages);
  const pct = Math.round(p * 100);
  const active = job.state === "queued" || job.state === "running";
  // Which stage is being worked on (or where it stopped), shown next to the pill.
  const at = job.stages.find((s) => s.state === (job.state === "failed" ? "failed" : "running"));
  const note = job.state === "running" || job.state === "failed" ? at && stageLabel(at.name) : null;
  return (
    <div className="job-badge" title={job.error ?? undefined}>
      <div className="job-line">
        <span className={`pill mono ${JOB_PILL[job.state] ?? ""}`}>{`${JOB_LABEL[job.state] ?? job.state}${job.state === "running" || ((job.state === "failed" || job.state === "cancelled") && pct > 0) ? ` · ${pct}%` : ""}`}</span>
        {note && <span className="muted small ellipsis">{note}</span>}
      </div>
      {active && (
        <div className="bar" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={pct}>
          <i style={{ width: `${p * 100}%` }} />
        </div>
      )}
    </div>
  );
}

export function JobDetail({ job, title, onRetry }: { job: Job; title?: string; onRetry?: () => void }) {
  const p = jobProgress(job.stages);
  const pct = Math.round(p * 100);
  const counted = job.stages.filter((s) => s.state !== "skipped");
  const done = counted.filter((s) => s.state === "succeeded").length;
  const current = job.stages.find((s) => s.state === "running");
  const barCls = job.state === "succeeded" ? "ok" : job.state === "failed" ? "bad" : job.state === "cancelled" ? "" : "accent";
  return (
    <section className={`card job-card ${job.state}`}>
      <div className="job-head">
        <div className="job-title">
          <h2>
            <span className={`status-dot ${job.state}`} />
            {title ?? "分析"}
            {JOB_PHRASE[job.state] ?? JOB_LABEL[job.state] ?? job.state}
          </h2>
          <p className="muted small">
            已完成 <span className="mono">{done} / {counted.length}</span> 个阶段
            {current && <> · 当前：{stageLabel(current.name)}</>}
          </p>
        </div>
        <div className="job-pct mono" aria-label={`总进度 ${pct}%`}>
          {pct}
          <small>%</small>
        </div>
      </div>
      <div className={`bar job-total ${barCls}`}>
        <i style={{ width: `${p * 100}%` }} />
      </div>
      <div className="stages">
        {job.stages.map((s, i) => {
          const sp = s.state === "succeeded" ? 1 : s.progress;
          return (
            <div key={s.name} className={`stage-row ${s.state}`}>
              <span className="step mono">{s.state === "succeeded" ? <Icon name="check" /> : i + 1}</span>
              <div className="body">
                <div className="stage-line">
                  <span className="name">{stageLabel(s.name)}</span>
                  <span className="spacer" />
                  <span className={`state small ${s.state === "failed" ? "error" : "muted"}`}>
                    {JOB_LABEL[s.state] ?? s.state}
                    {s.state === "running" && <span className="mono"> · {Math.round(sp * 100)}%</span>}
                  </span>
                </div>
                <div className={`bar ${s.state === "succeeded" ? "ok" : s.state === "failed" ? "bad" : s.state === "running" ? "accent" : ""}`}>
                  <i style={{ width: `${sp * 100}%` }} />
                </div>
                {s.error && <span className="error small">{s.error}</span>}
              </div>
            </div>
          );
        })}
      </div>
      {job.error && (
        <div className="notice bad job-error">
          <Icon name="alert" />
          <span>{job.error}</span>
        </div>
      )}
      {onRetry && (job.state === "failed" || job.state === "cancelled") && (
        <div className="job-retry">
          <button className="primary" onClick={onRetry}>
            <Icon name="refresh" />
            重新分析
          </button>
          <span className="muted small">从头重新跑一遍全部阶段；已做的人工修正会保留。</span>
        </div>
      )}
    </section>
  );
}
