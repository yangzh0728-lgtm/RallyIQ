import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../lib/api";
import { formatMs } from "../lib/corrections";
import type { Video } from "../lib/types";
import { JobBadge } from "../components/JobProgress";

export function Home() {
  const [videos, setVideos] = useState<Video[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [upload, setUpload] = useState<{ name: string; progress: number } | null>(null);
  const [over, setOver] = useState(false);
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

  const onFiles = async (files: FileList | null) => {
    const file = files?.[0];
    if (!file) return;
    if (!file.type.startsWith("video/") && !/\.(mp4|mov|m4v|webm|mkv)$/i.test(file.name)) {
      setError("请选择视频文件（mp4 / mov / webm）");
      return;
    }
    setUpload({ name: file.name, progress: 0 });
    try {
      const v = await api.uploadVideo(file, (p) => setUpload({ name: file.name, progress: p }));
      location.hash = `#/videos/${encodeURIComponent(v.video_id)}`;
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setUpload(null);
      refresh();
    }
  };

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
                <span>正在上传 {upload.name}</span>
                <span className="spacer" />
                <span className="mono">{Math.round(upload.progress * 100)}%</span>
              </div>
              <div className="bar">
                <i style={{ width: `${upload.progress * 100}%` }} />
              </div>
            </div>
          ) : (
            <>
              <div>拖入视频，或点击选择文件</div>
              <div className="muted small" style={{ marginTop: 6 }}>
                手机随手拍即可，不限机位和帧率；上传后自动开始分析
              </div>
            </>
          )}
          <input ref={input} type="file" accept="video/*" hidden onChange={(e) => void onFiles(e.target.files)} />
        </div>
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
        <h2>视频</h2>
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
              </tr>
            </thead>
            <tbody>
              {videos.map((v) => (
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
                    <JobBadge job={v.latest_job} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>
    </main>
  );
}
