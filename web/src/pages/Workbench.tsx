import { useCallback, useEffect, useMemo, useState } from "react";
import { CorrectionPanel, StatusPill } from "../components/CorrectionPanel";
import { FootworkPanel } from "../components/FootworkPanel";
import { HeatmapPanel } from "../components/HeatmapPanel";
import { JobDetail, useJob } from "../components/JobProgress";
import { MiniMap } from "../components/MiniMap";
import { QualityPanel } from "../components/QualityPanel";
import { StrokePanel } from "../components/StrokePanel";
import { Timeline } from "../components/Timeline";
import { VideoStage } from "../components/VideoStage";
import { api, type MatchFormatSetting } from "../lib/api";
import { PlaybackClock } from "../lib/clock";
import { rallyAt } from "../lib/corrections";
import { CORNER_NAMES } from "../lib/court";
import { download, exportCsv, exportJson } from "../lib/export";
import { FootIndex } from "../lib/footwork";
import { courtAt, DEFAULT_LAYERS, frameIndexAt, type Layers } from "../lib/overlay";
import { colorOf, labelOf } from "../lib/players";
import { playerIdsFor, type Correction, type CorrectionOp, type EventsResult, type FootworkResult, type Job, type OverlayData, type Point, type Video } from "../lib/types";

export function Workbench({ videoId, startMs = null }: { videoId: string; startMs?: number | null }) {
  const [video, setVideo] = useState<Video | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  useEffect(() => {
    api.getVideo(videoId).then(
      (v) => {
        setVideo(v);
        setJobId(v.latest_job?.job_id ?? null);
      },
      (e: Error) => setLoadError(e.message),
    );
  }, [videoId]);
  const { value: job, error: jobError } = useJob(jobId);
  const [startError, setStartError] = useState<string | null>(null);
  const start = () => {
    setStartError(null);
    api.startAnalysis(videoId).then(
      (j) => setJobId(j.job_id),
      (e: Error) => setStartError(e.message),
    );
  };

  if (loadError) return <Centered>无法加载视频：{loadError}</Centered>;
  if (!video) return <Centered>加载中…</Centered>;
  if (!jobId)
    return (
      <Centered>
        <div className="row">
          <span>这个视频还没有分析任务。</span>
          <button className="primary" onClick={start}>
            开始分析
          </button>
        </div>
        {startError && <p className="error small">启动失败：{startError}</p>}
      </Centered>
    );
  if (!job) return <Centered>{jobError ? `任务状态获取失败：${jobError}` : "获取任务状态…"}</Centered>;
  if (job.state !== "succeeded")
    return (
      <>
        <JobDetail job={job} onRetry={start} />
        {startError && <p className="error small" style={{ textAlign: "center" }}>启动失败：{startError}</p>}
      </>
    );
  return <Analysis key={job.job_id} video={video} startMs={startMs} onRerun={(j: Job) => setJobId(j.job_id)} />;
}

function Centered({ children }: { children: React.ReactNode }) {
  return (
    <main className="page">
      <div className="card">{children}</div>
    </main>
  );
}

function Analysis({ video, startMs, onRerun }: { video: Video; startMs: number | null; onRerun: (job: Job) => void }) {
  const [overlay, setOverlay] = useState<OverlayData | null>(null);
  const [events, setEvents] = useState<EventsResult | null>(null);
  const [corrections, setCorrections] = useState<Correction[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);
  const [hardExample, setHardExample] = useState(true);
  const [layers, setLayers] = useState<Layers>(DEFAULT_LAYERS);
  const [calibrating, setCalibrating] = useState(false);
  const [choosingCourt, setChoosingCourt] = useState(false);
  const [selRally, setSelRally] = useState<string | null>(null);
  const [selShot, setSelShot] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([api.getOverlay(video.video_id), api.getEvents(video.video_id), api.listCorrections(video.video_id)]).then(
      ([o, e, c]) => {
        setOverlay(o);
        setEvents(e);
        setCorrections(c);
      },
      (e: Error) => setError(e.message),
    );
  }, [video.video_id]);

  // Footwork is linked to the effective hits, so reload it whenever the events change (corrections).
  const [footwork, setFootwork] = useState<FootworkResult | null>(null);
  const [footworkError, setFootworkError] = useState<string | null>(null);
  useEffect(() => {
    if (!events) return;
    let live = true;
    api.getFootwork(video.video_id).then(
      (f) => {
        if (!live) return;
        setFootwork(f);
        setFootworkError(null);
      },
      (e: Error) => live && setFootworkError(e.message),
    );
    return () => {
      live = false;
    };
  }, [video.video_id, events]);
  const footIndex = useMemo(() => (footwork ? new FootIndex(footwork) : null), [footwork]);

  const duration = overlay?.video.duration_ms;
  const clock = useMemo(() => (duration ? new PlaybackClock(duration) : null), [duration]);
  useEffect(() => {
    if (clock && startMs != null) clock.seek(Math.max(0, startMs - 1000)); // evidence links: #/videos/<id>?t=<ms>
  }, [clock, startMs]);
  const fps = overlay?.video.fps ?? 30;

  const rallyNos = useMemo(() => new Map(events?.rallies.map((r, i) => [r.rally_id, i + 1]) ?? []), [events]);
  const rallyNo = useCallback((id: string) => rallyNos.get(id) ?? 0, [rallyNos]);

  // The current rally is the one under the playhead (null between rallies).
  useEffect(() => {
    if (!clock || !events) return;
    const sync = () => {
      const id = rallyAt(events.rallies, clock.ms)?.rally_id ?? null;
      setSelRally((cur) => (cur === id ? cur : id));
    };
    sync();
    return clock.subscribe(sync);
  }, [clock, events]);

  const submit = useCallback(
    async (op: CorrectionOp) => {
      setBusy(true);
      setActionError(null);
      try {
        const c = await api.postCorrection(video.video_id, { ...op, hard_example: hardExample });
        setCorrections((cs) => [...cs, c]);
        const ev = await api.getEvents(video.video_id);
        setEvents(ev);
        if (ev.corrections_skipped?.includes(c.correction_id)) setActionError("这条修正指向的击球/回合已不存在，后端已跳过。");
        if (op.op === "delete_shot") setSelShot(null);
        if (op.op === "add_shot") {
          const added = ev.shots.find((s) => s.source === "human" && s.frame === op.payload.frame);
          if (added) setSelShot(added.shot_id);
        }
      } catch (e) {
        setActionError(`保存失败：${(e as Error).message}`);
      } finally {
        setBusy(false);
      }
    },
    [video.video_id, hardExample],
  );

  const [formatSetting, setFormatSetting] = useState<MatchFormatSetting | null>(null);
  useEffect(() => {
    api.getMatchFormat(video.video_id).then(setFormatSetting, () => setFormatSetting(null));
  }, [video.video_id]);
  const changeFormat = async (f: MatchFormatSetting) => {
    try {
      setFormatSetting(f);
      onRerun(await api.setMatchFormat(video.video_id, f));
    } catch (e) {
      setActionError(`切换单/双打失败：${(e as Error).message}`);
    }
  };

  const onCalibrated = async (corners: Point[], otherCourts: Point[][] = []) => {
    setCalibrating(false);
    setChoosingCourt(false);
    try {
      onRerun(await api.setManualCalibration(video.video_id, corners, otherCourts));
    } catch (e) {
      setActionError(`标定失败：${(e as Error).message}`);
    }
  };

  const rally = events?.rallies.find((r) => r.rally_id === selRally);
  const rallyShots = useMemo(() => (events && rally ? events.shots.filter((s) => s.rally_id === rally.rally_id) : []), [events, rally]);
  const seekShot = useCallback(
    (id: string) => {
      setSelShot(id);
      const s = events?.shots.find((x) => x.shot_id === id);
      if (s && clock) {
        clock.pause();
        clock.seek(s.t_ms);
      }
    },
    [events, clock],
  );

  // Review queue: model shots / stroke types / rallies that still need a human look, in time order.
  const reviewQueue = useMemo(() => {
    if (!events) return [];
    const items: { t_ms: number; shot_id: string | null }[] = [];
    for (const s of events.shots) {
      const shotOpen = s.status === "needs_review" && s.source === "model";
      const typeOpen = s.stroke?.source === "model" && s.stroke.status !== "ok";
      if (shotOpen || typeOpen) items.push({ t_ms: s.t_ms, shot_id: s.shot_id });
    }
    for (const r of events.rallies) if (r.status === "needs_review" && r.source === "model") items.push({ t_ms: r.start_ms, shot_id: null });
    return items.sort((a, b) => a.t_ms - b.t_ms);
  }, [events]);
  const nextReview = useCallback(
    (dir: 1 | -1) => {
      if (!clock || !reviewQueue.length) return;
      const t = clock.ms;
      const item =
        dir > 0
          ? (reviewQueue.find((x) => x.t_ms > t + 1) ?? reviewQueue[0])
          : ([...reviewQueue].reverse().find((x) => x.t_ms < t - 1) ?? reviewQueue[reviewQueue.length - 1]);
      if (item.shot_id) seekShot(item.shot_id);
      else {
        clock.pause();
        clock.seek(item.t_ms);
      }
    },
    [clock, reviewQueue, seekShot],
  );
  const [showKeys, setShowKeys] = useState(false);

  // Keyboard shortcuts.
  useEffect(() => {
    if (!clock || !events) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement || e.target instanceof HTMLTextAreaElement) return;
      const t = clock.ms;
      const frameMs = 1000 / fps;
      const shotNear = (dir: 1 | -1) => (dir > 0 ? events.shots.find((x) => x.t_ms > t + 1) : [...events.shots].reverse().find((x) => x.t_ms < t - 1));
      const rallyNear = (dir: 1 | -1) => (dir > 0 ? events.rallies.find((x) => x.start_ms > t + 1) : [...events.rallies].reverse().find((x) => x.start_ms < t - 1));
      const actions: Record<string, () => void> = {
        " ": () => clock.toggle(),
        ArrowLeft: () => (clock.pause(), clock.seek(t - frameMs)),
        ArrowRight: () => (clock.pause(), clock.seek(t + frameMs)),
        j: () => {
          const s = shotNear(-1);
          if (s) seekShot(s.shot_id);
        },
        k: () => {
          const s = shotNear(1);
          if (s) seekShot(s.shot_id);
        },
        "[": () => {
          const r = rallyNear(-1);
          if (r) clock.seek(r.start_ms);
        },
        "]": () => {
          const r = rallyNear(1);
          if (r) clock.seek(r.start_ms);
        },
        n: () => nextReview(1),
        N: () => nextReview(-1),
        "?": () => setShowKeys((v) => !v),
        Escape: () => setShowKeys(false),
      };
      const fn = actions[e.key];
      if (fn) {
        e.preventDefault();
        fn();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [clock, events, fps, seekShot, nextReview]);

  const doExport = async (format: "json" | "csv") => {
    const url = api.exportUrl(video.video_id, format);
    if (url) {
      // Backend sets Content-Disposition, so a plain navigation downloads the file.
      window.location.href = url;
      return;
    }
    if (!events) return;
    if (format === "json") {
      const raw = await api.getRawEvents(video.video_id);
      download(`${video.video_id}_events.json`, exportJson(video, events, raw, corrections), "application/json");
    } else download(`${video.video_id}_shots.csv`, exportCsv(events), "text/csv");
  };

  if (error) return <Centered>加载分析结果失败：{error}</Centered>;
  if (!overlay || !events || !clock) return <Centered>加载分析结果…</Centered>;

  const firstFrame = overlay.frames[frameIndexAt(overlay.frames, 0)]?.frame ?? 0;
  const court = courtAt(overlay.court, firstFrame) ?? overlay.court[0];
  const reviewShots = events.shots.filter((s) => s.status === "needs_review" && s.source === "model").length;
  const reviewRallies = events.rallies.filter((r) => r.status === "needs_review" && r.source === "model").length;
  const reviewTypes = events.shots.filter((s) => s.stroke?.source === "model" && s.stroke.status !== "ok").length;
  const toggle = (k: keyof Omit<Layers, "opacity">) => setLayers((l) => ({ ...l, [k]: !l[k] }));
  const meta = overlay.video;

  return (
    <div className="wb">
      <aside className="wb-left">
        <div className="panel small">
          <h3 title={video.filename} style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
            {video.filename}
          </h3>
          <Stat label="规格">
            {meta.width}×{meta.height} · {Math.round(meta.fps)}fps{meta.variable_frame_rate ? "（可变帧率）" : ""}
          </Stat>
          <Stat label="回合 / 击球">
            {events.rallies.length} / {events.shots.length}
          </Stat>
          <Stat label="待复核 回合 / 击球">
            <span style={{ color: reviewShots + reviewRallies ? "var(--warn)" : undefined }}>
              {reviewRallies} / {reviewShots}
            </span>
          </Stat>
          <Stat label="击球类型 待复核 / 不可判定">
            <span style={{ color: reviewTypes ? "var(--warn)" : undefined }}>
              {events.shots.filter((s) => s.stroke?.source === "model" && s.stroke.status === "needs_review").length} /{" "}
              {events.shots.filter((s) => s.stroke?.source === "model" && s.stroke.status === "undeterminable").length}
            </span>
          </Stat>
          <Stat label="已做修正">{corrections.length}</Stat>
          <div className="row" style={{ marginTop: 6 }}>
            <button
              disabled={!reviewQueue.length}
              title="跳到当前时间之后下一个待复核的回合、击球或击球类型（快捷键 n，Shift+N 往回）"
              onClick={() => nextReview(1)}
            >
              {reviewQueue.length ? `下一个待复核（剩 ${reviewQueue.length}）` : "没有待复核项"}
            </button>
            <span className="spacer" />
            <button className="small" title="快捷键（?）" onClick={() => setShowKeys(true)}>
              快捷键
            </button>
          </div>
          <div className="row">
            <span className="muted">比赛类型</span>
            <span className="spacer" />
            <span className="pill">{overlay.match_format === "doubles" ? "双打 · 4 人" : "单打 · 2 人"}</span>
            <select
              value={formatSetting ?? "auto"}
              disabled={formatSetting === null}
              title="自动识别不对时手动指定；会从球员识别开始重新分析"
              onChange={(e) => void changeFormat(e.target.value as MatchFormatSetting)}
            >
              <option value="auto">自动识别</option>
              <option value="singles">强制单打</option>
              <option value="doubles">强制双打</option>
            </select>
          </div>
          <div className="row">
            <span className="muted">球场标定</span>
            <span className="spacer" />
            {court ? <StatusPill status={court.status} /> : <span className="pill needs_review">未标定</span>}
            {overlay.court.length > 1 && <span className="muted">{overlay.court.length} 段</span>}
          </div>
          <button style={{ marginTop: 8, width: "100%" }} className={calibrating ? "active" : ""} onClick={() => setCalibrating((c) => !c)}>
            {calibrating ? "取消点选" : "球场线不对？手动点选四角"}
          </button>
          {calibrating && <div className="muted" style={{ marginTop: 4 }}>按 {CORNER_NAMES.join(" → ")} 的顺序点击外框四角；提交后会从标定开始重新分析。</div>}
          {!!court?.other_courts?.length && (
            <>
              <div className="muted" style={{ marginTop: 8 }}>
                画面里还有 {court.other_courts.length} 块场地（虚线框），它们的球员和球已被排除。
              </div>
              <button
                style={{ marginTop: 4, width: "100%" }}
                className={choosingCourt ? "active" : ""}
                onClick={() => {
                  setCalibrating(false);
                  setChoosingCourt((c) => !c);
                }}
              >
                {choosingCourt ? "取消选择" : "分析的不是这块场地？换一块"}
              </button>
            </>
          )}
        </div>
        <QualityPanel videoId={video.video_id} />
        <HeatmapPanel overlay={overlay} rally={rally} rallyNo={rally ? rallyNo(rally.rally_id) : null} courtStatus={court?.status} />
        <StrokePanel shots={events.shots} rally={rally} rallyNo={rally ? rallyNo(rally.rally_id) : null} clock={clock} />
      </aside>

      <section className="wb-center">
        <div className="controls">
          <button className={layers.ai ? "active" : ""} onClick={() => toggle("ai")} title="原画 / AI 标注切换">
            {layers.ai ? "AI 标注" : "原画"}
          </button>
          {(
            [
              ["skeleton", "骨架"],
              ["shuttle", "羽毛球"],
              ["court", "球场线"],
              ["bbox", "检测框"],
              ["feet", "脚步"],
            ] as const
          ).map(([k, label]) => (
            <label key={k}>
              <input type="checkbox" checked={layers[k]} disabled={!layers.ai} onChange={() => toggle(k)} />
              {label}
            </label>
          ))}
          <label title="隐藏原视频画面，只在深色背景上显示识别元素">
            <input type="checkbox" checked={layers.abstract} disabled={!layers.ai} onChange={() => toggle("abstract")} />
            纯抽象
          </label>
          <label>
            透明度
            <input type="range" min={0.2} max={1} step={0.05} value={layers.opacity} onChange={(e) => setLayers((l) => ({ ...l, opacity: Number(e.target.value) }))} />
          </label>
          <span className="spacer" />
          <span className="legend">
            {playerIdsFor(overlay.match_format).map((id) => (
              <span key={id} className="row" style={{ gap: 4 }}>
                <span className="dot" style={{ background: colorOf(id) }} />
                {labelOf(id)}
              </span>
            ))}
            <span className="row" style={{ gap: 4 }}>
              <span className="dot" style={{ background: "#facc15" }} />球：实线=检测，虚线=插值
            </span>
          </span>
        </div>
        <VideoStage overlay={overlay} videoUrl={api.videoFileUrl(video.video_id)} clock={clock} layers={layers} calibrating={calibrating} onCalibrated={onCalibrated} choosingCourt={choosingCourt} onCourtChosen={onCalibrated} feet={footIndex} />
      </section>

      <aside className="wb-right">
        <MiniMap overlay={overlay} clock={clock} />
        <FootworkPanel
          footwork={footwork}
          error={footworkError}
          players={playerIdsFor(overlay.match_format)}
          rally={rally}
          rallyNo={rally ? rallyNo(rally.rally_id) : null}
          selectedShotId={selShot}
          clock={clock}
        />
        <CorrectionPanel
          rally={rally}
          rallyNo={rally ? rallyNo(rally.rally_id) : null}
          shots={rallyShots}
          selectedShotId={selShot}
          onSelectShot={seekShot}
          clock={clock}
          fps={fps}
          submit={submit}
          busy={busy}
          hardExample={hardExample}
          setHardExample={setHardExample}
          corrections={corrections}
          error={actionError}
          matchFormat={events.match_format ?? overlay.match_format}
        />
        <div className="panel">
          <h3>赛后报告</h3>
          <div className="muted small" style={{ marginBottom: 6 }}>回合、跑动、覆盖、得失分统计和 AI 教练点评（基于修正后的数据）。</div>
          <a href={`#/videos/${encodeURIComponent(video.video_id)}/report`}>
            <button className="primary" style={{ width: "100%" }}>查看赛后报告 →</button>
          </a>
        </div>
        <div className="panel">
          <h3>导出</h3>
          <div className="row">
            <button onClick={() => void doExport("json")}>JSON（含原始预测和修正）</button>
            <button onClick={() => void doExport("csv")}>CSV（逐拍）</button>
          </div>
          <div className="row" style={{ marginTop: 6 }}>
            <button
              disabled={!rally || !api.clipUrl(video.video_id, rally.rally_id)}
              title={api.clipUrl(video.video_id, "x") ? "导出当前回合的视频片段（前后各留 1 秒）" : "演示模式没有原视频"}
              onClick={() => {
                const url = rally && api.clipUrl(video.video_id, rally.rally_id);
                if (url) window.location.href = url;
              }}
            >
              {rally ? `第 ${rallyNo(rally.rally_id)} 回合片段（MP4）` : "回合片段（先播放到某个回合）"}
            </button>
          </div>
        </div>
      </aside>

      {showKeys && <ShortcutHelp onClose={() => setShowKeys(false)} />}
      <footer className="wb-timeline">
        <Timeline
          durationMs={meta.duration_ms}
          rallies={events.rallies}
          shots={events.shots}
          clock={clock}
          selectedRallyId={selRally}
          selectedShotId={selShot}
          fps={fps}
          rallyNo={rallyNo}
          footwork={layers.feet ? footwork?.events : undefined}
          onSelectRally={(id) => {
            const r = events.rallies.find((x) => x.rally_id === id);
            if (r) clock.seek(r.start_ms);
          }}
          onSelectShot={seekShot}
        />
      </footer>
    </div>
  );
}

function Stat({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="row">
      <span className="muted">{label}</span>
      <span className="spacer" />
      <span className="mono">{children}</span>
    </div>
  );
}

const SHORTCUTS: [string, string][] = [
  ["空格", "播放 / 暂停"],
  ["← / →", "上一帧 / 下一帧"],
  ["j / k", "上一拍 / 下一拍"],
  ["[ / ]", "上一个 / 下一个回合"],
  ["n / Shift+N", "下一个 / 上一个待复核项"],
  ["?", "显示或关闭这个面板"],
];

function ShortcutHelp({ onClose }: { onClose: () => void }) {
  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="card modal" role="dialog" aria-label="快捷键" onClick={(e) => e.stopPropagation()}>
        <div className="row" style={{ marginBottom: 8 }}>
          <h2 style={{ margin: 0 }}>快捷键</h2>
          <span className="spacer" />
          <button className="small" onClick={onClose}>
            关闭
          </button>
        </div>
        <table>
          <tbody>
            {SHORTCUTS.map(([k, v]) => (
              <tr key={k}>
                <td>
                  <kbd>{k}</kbd>
                </td>
                <td>{v}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="muted small">在输入框里打字时快捷键不生效。</p>
      </div>
    </div>
  );
}
