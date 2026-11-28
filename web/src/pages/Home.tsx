import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "../lib/api";
import { formatMs } from "../lib/corrections";
import { isVideoFile, VIDEO_EXT } from "../lib/files";
import type { Video } from "../lib/types";
import { HardExamples } from "../components/HardExamples";
import { JobBadge } from "../components/JobProgress";
import { CourtGlyph, Empty, Icon, Page, PageHeader, StatTile, type IconName } from "../components/ui";

type Sort = "new" | "old" | "name" | "long";

/** What happens to a video after it is uploaded, shown beside the drop zone. */
const FLOW: { icon: IconName; title: string; text: string }[] = [
  { icon: "upload", title: "上传视频", text: "手机随手拍即可，不限机位和帧率" },
  { icon: "sparkles", title: "自动分析", text: "球场标定、球员骨架、羽毛球轨迹、回合与击球" },
  { icon: "edit", title: "复核纠错", text: "逐拍确认或修改；修正只追加，原始预测始终保留" },
  { icon: "report", title: "赛后报告与档案", text: "统计、AI 教练点评、长期趋势" },
];

const pad2 = (n: number) => String(n).padStart(2, "0");

/** List duration: m:ss (h:mm:ss from one hour); the exact value stays in the tooltip. */
function clock(ms: number): string {
  const s = Math.round(Math.max(0, ms) / 1000);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  return h ? `${h}:${pad2(m)}:${pad2(s % 60)}` : `${m}:${pad2(s % 60)}`;
}

/** Total footage: minutes below one hour, h:mm from there. */
function footage(ms: number): { value: string; unit: string } {
  const min = ms / 60_000;
  if (min < 60) return { value: min > 0 && min < 10 ? (Math.round(min * 10) / 10).toString() : String(Math.round(min)), unit: "分钟" };
  const total = Math.round(min);
  return { value: `${Math.floor(total / 60)}:${pad2(total % 60)}`, unit: "小时" };
}

function fileSize(bytes: number): string {
  const mb = bytes / (1024 * 1024);
  if (mb < 1) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  return mb >= 1024 ? `${(mb / 1024).toFixed(1)} GB` : `${mb < 10 ? mb.toFixed(1) : Math.round(mb)} MB`;
}

const when = (iso: string) => new Date(iso).toLocaleString("zh-CN", { month: "numeric", day: "numeric", hour: "2-digit", minute: "2-digit" });

export function Home() {
  const [videos, setVideos] = useState<Video[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [upload, setUpload] = useState<{ name: string; progress: number; index: number; total: number } | null>(null);
  const [failed, setFailed] = useState<string[]>([]);
  const [over, setOver] = useState(false);
  const [query, setQuery] = useState("");
  const [sort, setSort] = useState<Sort>("new");
  const input = useRef<HTMLInputElement>(null);

  const refresh = useCallback(() => {
    api.listVideos().then(
      (v) => {
        setVideos(v);
        setError(null);
      },
      (e: Error) => setError(e.message),
    );
  }, []);
  useEffect(refresh, [refresh]);

  // Every file is uploaded in turn; with one file the workbench opens, with several the list stays.
  const onFiles = async (files: FileList | null) => {
    const all = Array.from(files ?? []);
    if (!all.length) return;
    const ok = all.filter(isVideoFile);
    const skipped = all.filter((f) => !isVideoFile(f)).map((f) => f.name);
    setFailed([]);
    setError(skipped.length ? `已跳过不支持的文件：${skipped.join("、")}（支持 ${VIDEO_EXT.join(" ")}）` : null);
    const errors: string[] = [];
    let last: string | null = null;
    for (const [i, file] of ok.entries()) {
      setUpload({ name: file.name, progress: 0, index: i + 1, total: ok.length });
      try {
        const v = await api.uploadVideo(file, (p) => setUpload({ name: file.name, progress: p, index: i + 1, total: ok.length }));
        last = v.video_id;
      } catch (e) {
        errors.push(`${file.name}：${(e as Error).message}`);
      }
      refresh();
    }
    setUpload(null);
    setFailed(errors);
    if (ok.length === 1 && last && !errors.length) location.hash = `#/videos/${encodeURIComponent(last)}`;
  };

  const shown = useMemo(() => {
    const q = query.trim().toLowerCase();
    const list = (videos ?? []).filter((v) => !q || v.filename.toLowerCase().includes(q));
    const by: Record<Sort, (a: Video, b: Video) => number> = {
      new: (a, b) => b.created_at.localeCompare(a.created_at),
      old: (a, b) => a.created_at.localeCompare(b.created_at),
      name: (a, b) => a.filename.localeCompare(b.filename, "zh"),
      long: (a, b) => (b.meta?.duration_ms ?? 0) - (a.meta?.duration_ms ?? 0),
    };
    return [...list].sort(by[sort]);
  }, [videos, query, sort]);

  // Overview numbers, all derived from the list that is already loaded.
  const stats = useMemo(() => {
    if (!videos) return null;
    let done = 0;
    let active = 0;
    let failedJobs = 0;
    let ms = 0;
    let latest: string | null = null;
    for (const v of videos) {
      const st = v.latest_job?.state;
      if (st === "succeeded") done++;
      else if (st === "running" || st === "queued") active++;
      else if (st === "failed") failedJobs++;
      ms += v.meta?.duration_ms ?? 0;
      if (!latest || v.created_at > latest) latest = v.created_at;
    }
    return { total: videos.length, done, active, failedJobs, footage: footage(ms), latest };
  }, [videos]);

  const pick = () => input.current?.click();
  // Shown in the overview tiles until the list has loaded.
  const pending = <span className="muted">—</span>;
  const filtering = !!videos && shown.length !== videos.length;

  return (
    <Page
      header={
        <PageHeader
          title="比赛库"
          crumbs={[{ label: "工作区" }]}
          actions={
            <button className="primary" onClick={pick}>
              <Icon name="upload" />
              上传视频
            </button>
          }
        />
      }
    >
      <div className="stat-strip">
        <StatTile
          icon="video"
          label="视频总数"
          value={stats ? stats.total : pending}
          unit={stats ? "个" : undefined}
          sub={stats?.latest ? `最近上传 ${when(stats.latest)}` : stats ? "还没有上传" : "\u00a0"}
        />
        <StatTile
          icon="check"
          label="已完成分析"
          value={stats ? stats.done : pending}
          unit={stats ? `/ ${stats.total}` : undefined}
          sub={stats ? (stats.done > 0 ? `${stats.done} 场可打开赛后报告` : "分析完成后可打开赛后报告") : "\u00a0"}
        />
        <StatTile
          icon="refresh"
          label="分析中"
          value={stats ? stats.active : pending}
          unit={stats ? "个" : undefined}
          sub={
            stats ? (
              stats.failedJobs > 0 ? (
                <span className="row nowrap" style={{ gap: 6 }}>
                  <span className="status-dot bad" />
                  {stats.failedJobs} 个分析失败，可进入视频重新分析
                </span>
              ) : stats.active > 0 ? (
                "含排队中的任务，完成后列表自动更新"
              ) : (
                "没有进行中或排队的任务"
              )
            ) : (
              "\u00a0"
            )
          }
        />
        <StatTile icon="clock" label="累计时长" value={stats ? stats.footage.value : pending} unit={stats ? stats.footage.unit : undefined} sub={stats ? "已上传素材" : "\u00a0"} />
      </div>

      <section className="card upload-card">
        <div className="upload-main">
          <div className="card-head">
            <h2>上传比赛或训练视频</h2>
            <span className="sub">上传后自动开始分析</span>
          </div>
          <div
            className={`dropzone ${over ? "over" : ""}${upload ? " busy" : ""}`}
            role="button"
            tabIndex={0}
            aria-label="上传视频：拖入文件，或点击选择"
            onClick={pick}
            onKeyDown={(e) => {
              if (e.target !== e.currentTarget || (e.key !== "Enter" && e.key !== " ")) return;
              e.preventDefault();
              pick();
            }}
            onDragOver={(e) => {
              e.preventDefault();
              setOver(true);
            }}
            onDragLeave={() => setOver(false)}
            onDrop={(e) => {
              e.preventDefault();
              setOver(false);
              void onFiles(e.dataTransfer.files);
            }}
          >
            {upload ? (
              <div className="upload-progress">
                <div className="row nowrap">
                  <b className="ellipsis" title={upload.name}>
                    {upload.name}
                  </b>
                  <span className="spacer" />
                  <span className="mono pct">{Math.round(upload.progress * 100)}%</span>
                </div>
                <div className="bar accent">
                  <i style={{ width: `${upload.progress * 100}%` }} />
                </div>
                <div className="muted small">
                  正在上传{upload.total > 1 ? `（${upload.index}/${upload.total}）` : ""}，完成后自动开始分析
                </div>
              </div>
            ) : (
              <>
                <span className="glyph">
                  <Icon name="upload" className="lg" />
                </span>
                <b>拖入视频，或点击选择文件</b>
                <div className="muted small">可以一次选多个；手机随手拍即可，不限机位和帧率</div>
                <div className="formats">
                  {VIDEO_EXT.map((e) => (
                    <span key={e} className="chip mono">
                      {e}
                    </span>
                  ))}
                </div>
              </>
            )}
          </div>
          <input
            ref={input}
            type="file"
            accept={["video/*", ...VIDEO_EXT].join(",")}
            multiple
            hidden
            onChange={(e) => {
              void onFiles(e.target.files);
              e.target.value = "";
            }}
          />
        </div>
        <div className="upload-flow">
          <div className="eyebrow">从视频到报告</div>
          <div className="pipeline">
            {FLOW.map((f, i) => (
              <div key={f.title} className="step">
                <div className="n">
                  <Icon name={f.icon} />
                </div>
                <div>
                  <b>
                    <i className="idx mono">{i + 1}</i>
                    {f.title}
                  </b>
                  <span>{f.text}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
        {(failed.length > 0 || error) && (
          <div className="upload-msgs">
            {failed.map((f) => (
              <div key={f} className="notice warn">
                <Icon name="alert" />
                <span>上传失败 · {f}</span>
              </div>
            ))}
            {error && (
              <p className="error small upload-error" role="alert">
                <Icon name="alert" />
                <span>
                  {error}
                  {api.mode === "live" && (
                    <>
                      {" "}
                      · 后端没启动？可以先用 <a href="?mock#/">演示模式</a> 查看界面。
                    </>
                  )}
                </span>
              </p>
            )}
          </div>
        )}
      </section>

      <section className="card videos-card">
        <div className="card-head">
          <h2>全部视频</h2>
          {videos && videos.length > 0 && (
            <span className="sub mono">{filtering ? `${shown.length} / ${videos.length} 个` : `${videos.length} 个`}</span>
          )}
          <span className="spacer" />
          {videos && videos.length > 1 && (
            <>
              <div className="search-field">
                <Icon name="search" />
                <input type="search" placeholder="按文件名搜索" value={query} onChange={(e) => setQuery(e.target.value)} aria-label="搜索视频" />
              </div>
              <select value={sort} onChange={(e) => setSort(e.target.value as Sort)} aria-label="排序">
                <option value="new">最新上传</option>
                <option value="old">最早上传</option>
                <option value="name">文件名</option>
                <option value="long">时长最长</option>
              </select>
            </>
          )}
        </div>
        {!videos ? (
          error ? (
            <Empty icon="alert" title="无法加载视频列表">
              原因见上方的提示，后端恢复后刷新页面即可。
            </Empty>
          ) : (
            <div className="video-list" role="status" aria-busy="true" aria-label="正在加载视频列表">
              {[0, 1, 2].map((i) => (
                <div key={i} className="video-row skeleton">
                  <span className="skel thumb" />
                  <span className="skel-lines">
                    <span className="skel" style={{ width: `${46 - i * 8}%` }} />
                    <span className="skel" style={{ width: "28%" }} />
                  </span>
                </div>
              ))}
            </div>
          )
        ) : videos.length === 0 ? (
          <Empty icon="video" title="还没有视频">
            把比赛或训练视频拖到上方，或点右上角的“上传视频”，上传后自动开始分析。
            {api.mode === "live" && (
              <>
                也可以先用 <a href="?mock#/">演示模式</a> 看看示例比赛。
              </>
            )}
          </Empty>
        ) : !shown.length ? (
          <Empty icon="search" title="没有匹配的视频">
            没有文件名包含「{query}」的视频。
          </Empty>
        ) : (
          <div className="video-list">
            {shown.map((v) => {
              const href = `#/videos/${encodeURIComponent(v.video_id)}`;
              return (
                <div key={v.video_id} className="video-row">
                  <a className="video-thumb" href={href} tabIndex={-1} aria-hidden="true">
                    <CourtGlyph />
                    <span className="play">
                      <Icon name="play" />
                    </span>
                  </a>
                  <div className="info">
                    <a className="title" href={href} title={v.filename}>
                      {v.filename}
                    </a>
                    <div className="meta">
                      {v.meta && (
                        <>
                          <span className="mono">
                            {v.meta.width}×{v.meta.height}
                          </span>
                          <span className="mono">
                            {Math.round(v.meta.fps)}fps{v.meta.variable_frame_rate ? "（可变帧率）" : ""}
                          </span>
                        </>
                      )}
                      <span className="mono" title={new Date(v.created_at).toLocaleString("zh-CN")}>
                        {when(v.created_at)} 上传
                      </span>
                      {v.size_bytes > 0 && <span className="mono opt">{fileSize(v.size_bytes)}</span>}
                    </div>
                  </div>
                  <div className="dur mono" title={v.meta ? `时长 ${formatMs(v.meta.duration_ms)}` : "时长未知"}>
                    <Icon name="clock" />
                    {v.meta ? clock(v.meta.duration_ms) : "—"}
                  </div>
                  <JobBadge job={v.latest_job} onDone={refresh} />
                  <div className="actions">
                    <a className="btn small" href={href}>
                      打开
                    </a>
                    {v.latest_job?.state === "succeeded" && (
                      <a className="btn small" href={`${href}/report`}>
                        报告
                      </a>
                    )}
                    <button
                      className="ghost icon-btn small danger"
                      title="删除视频、分析结果和修正记录"
                      aria-label={`删除「${v.filename}」`}
                      onClick={() => {
                        if (!confirm(`删除「${v.filename}」及其全部分析结果？此操作不能撤销。`)) return;
                        api.deleteVideo(v.video_id).then(refresh, (e: Error) => setError(e.message));
                      }}
                    >
                      <Icon name="trash" />
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </section>
      <HardExamples />
    </Page>
  );
}
