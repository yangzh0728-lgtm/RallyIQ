import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "../lib/api";
import { formatMs } from "../lib/corrections";
import { isVideoFile, VIDEO_EXT } from "../lib/files";
import type { Video } from "../lib/types";
import { HardExamples } from "../components/HardExamples";
import { JobBadge } from "../components/JobProgress";

type Sort = "new" | "old" | "name" | "long";

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

  return (
    <main className="page">
      <section className="card">
        <h2>上传比赛 / 训练视频</h2>
        <div
          className={`dropzone ${over ? "over" : ""}`}
          onClick={() => input.current?.click()}
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
            <div style={{ maxWidth: 420, margin: "0 auto" }}>
              <div className="row small">
                <span>
                  正在上传{upload.total > 1 ? `（${upload.index}/${upload.total}）` : ""} {upload.name}
                </span>
                <span className="spacer" />
                <span className="mono">{Math.round(upload.progress * 100)}%</span>
              </div>
              <div className="bar">
                <i style={{ width: `${upload.progress * 100}%` }} />
              </div>
            </div>
          ) : (
            <>
              <div>拖入视频，或点击选择文件（可以一次选多个）</div>
              <div className="muted small" style={{ marginTop: 6 }}>
                手机随手拍即可，不限机位和帧率；支持 {VIDEO_EXT.join(" ")}，上传后自动开始分析
              </div>
            </>
          )}
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
        {failed.length > 0 && (
          <ul className="error small">
            {failed.map((f) => (
              <li key={f}>上传失败 · {f}</li>
            ))}
          </ul>
        )}
        {error && (
          <p className="error small">
            {error}
            {api.mode === "live" && (
              <>
                {" "}
                · 后端没启动？可以先用 <a href="?mock#/">演示模式</a> 查看界面。
              </>
            )}
          </p>
        )}
      </section>

      <section className="card">
        <div className="row" style={{ marginBottom: 10 }}>
          <h2 style={{ margin: 0 }}>视频</h2>
          {videos && videos.length > 0 && <span className="muted small">{videos.length} 个</span>}
          <span className="spacer" />
          {videos && videos.length > 1 && (
            <>
              <input type="search" placeholder="按文件名搜索" value={query} onChange={(e) => setQuery(e.target.value)} aria-label="搜索视频" />
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
          <p className="muted">{error ? "无法加载" : "加载中…"}</p>
        ) : videos.length === 0 ? (
          <p className="muted">还没有视频。</p>
        ) : (
          <table>
            <thead>
              <tr>
                <th>文件</th>
                <th>时长</th>
                <th>规格</th>
                <th>上传时间</th>
                <th>分析状态</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {shown.map((v) => (
                <tr key={v.video_id}>
                  <td>
                    <a href={`#/videos/${encodeURIComponent(v.video_id)}`}>{v.filename}</a>
                  </td>
                  <td className="mono">{v.meta ? formatMs(v.meta.duration_ms) : "—"}</td>
                  <td className="muted small">
                    {v.meta ? `${v.meta.width}×${v.meta.height} · ${Math.round(v.meta.fps)}fps${v.meta.variable_frame_rate ? "（可变帧率）" : ""}` : "—"}
                  </td>
                  <td className="muted small">{new Date(v.created_at).toLocaleString()}</td>
                  <td>
                    <JobBadge job={v.latest_job} onDone={refresh} />
                  </td>
                  <td className="row" style={{ flexWrap: "nowrap" }}>
                    {v.latest_job?.state === "succeeded" && (
                      <a className="small" href={`#/videos/${encodeURIComponent(v.video_id)}/report`}>
                        报告
                      </a>
                    )}
                    <button
                      className="small danger"
                      title="删除视频、分析结果和修正记录"
                      onClick={() => {
                        if (!confirm(`删除「${v.filename}」及其全部分析结果？此操作不能撤销。`)) return;
                        api.deleteVideo(v.video_id).then(refresh, (e: Error) => setError(e.message));
                      }}
                    >
                      删除
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        {videos && videos.length > 0 && !shown.length && <p className="muted">没有文件名包含「{query}」的视频。</p>}
      </section>
      <HardExamples />
    </main>
  );
}
