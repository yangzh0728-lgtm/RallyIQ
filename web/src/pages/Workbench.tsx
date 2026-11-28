import { useCallback, useEffect, useMemo, useState, type ReactNode } from "react";
import { CorrectionPanel, StatusPill } from "../components/CorrectionPanel";
import { HeatmapPanel } from "../components/HeatmapPanel";
import { JobDetail, useJob } from "../components/JobProgress";
import { MiniMap } from "../components/MiniMap";
import { QualityPanel } from "../components/QualityPanel";
import { StrokePanel } from "../components/StrokePanel";
import { Timeline } from "../components/Timeline";
import { Empty, Icon, Page, PageHeader, VideoTabs, type Crumb } from "../components/ui";
import { VideoStage } from "../components/VideoStage";
import { api, type MatchFormatSetting } from "../lib/api";
import { PlaybackClock } from "../lib/clock";
import { rallyAt } from "../lib/corrections";
import { CORNER_NAMES } from "../lib/court";
import { download, exportCsv, exportJson } from "../lib/export";
import { courtAt, DEFAULT_LAYERS, frameIndexAt, type Layers } from "../lib/overlay";
import { colorOf, labelOf } from "../lib/players";
import { playerIdsFor, type Correction, type CorrectionOp, type EventsResult, type Job, type OverlayData, type Point, type Video } from "../lib/types";

const CRUMBS: Crumb[] = [{ label: "比赛库", href: "#/" }];

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

  if (loadError)
    return (
      <Shell video={null}>
        <Failed title="无法加载视频">{loadError}</Failed>
      </Shell>
    );
  if (!video)
    return (
      <Shell video={null}>
        <Loading>加载中…</Loading>
      </Shell>
    );
  if (!jobId)
    return (
      <Shell video={video}>
        <section className="card">
          <div className="empty">
            <span className="glyph">
              <Icon name="sparkles" className="lg" />
            </span>
            <b>这个视频还没有分析任务</b>
            <p className="small">开始后会依次完成球场标定、球员与羽毛球跟踪、回合与击球识别，完成后在这里回放和纠错。</p>
            <button className="primary" onClick={start}>
              <Icon name="play" />
              开始分析
            </button>
            {startError && <p className="error small">启动失败：{startError}</p>}
          </div>
        </section>
      </Shell>
    );
  if (!job)
    return (
      <Shell video={video}>{jobError ? <Failed title="任务状态获取失败">{jobError}</Failed> : <Loading>获取任务状态…</Loading>}</Shell>
    );
  if (job.state !== "succeeded")
    return (
      <Shell video={video}>
        <JobDetail job={job} onRetry={start} />
        {startError && (
          <p className="error small" style={{ textAlign: "center" }}>
            启动失败：{startError}
          </p>
        )}
      </Shell>
    );
  return <Analysis key={job.job_id} video={video} startMs={startMs} onRerun={(j: Job) => setJobId(j.job_id)} />;
}

/** Page frame for everything shown before the analysis desk is ready. */
function Shell({ video, children }: { video: Video | null; children: ReactNode }) {
  return (
    <Page
      className="narrow"
      header={<PageHeader crumbs={CRUMBS} title={video?.filename ?? "分析工作台"} center={video && <VideoTabs videoId={video.video_id} active="workbench" />} />}
    >
      {children}
    </Page>
  );
}

function Loading({ children }: { children: ReactNode }) {
  return (
    <section className="card wb-loading" role="status">
      {children}
    </section>
  );
}

function Failed({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="card" role="alert">
      <Empty icon="alert" title={title}>
        {children}
      </Empty>
    </section>
  );
}

const LAYER_TOGGLES = [
  ["skeleton", "骨架"],
  ["shuttle", "羽毛球"],
  ["court", "球场线"],
  ["bbox", "检测框"],
] as const;

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

  if (error)
    return (
      <Shell video={video}>
        <Failed title="加载分析结果失败">{error}</Failed>
      </Shell>
    );
  if (!overlay || !events || !clock)
    return (
      <Shell video={video}>
        <Loading>加载分析结果…</Loading>
      </Shell>
    );

  const firstFrame = overlay.frames[frameIndexAt(overlay.frames, 0)]?.frame ?? 0;
  const court = courtAt(overlay.court, firstFrame) ?? overlay.court[0];
  const reviewShots = events.shots.filter((s) => s.status === "needs_review" && s.source === "model").length;
  const reviewRallies = events.rallies.filter((r) => r.status === "needs_review" && r.source === "model").length;
  const reviewTypes = events.shots.filter((s) => s.stroke?.source === "model" && s.stroke.status !== "ok").length;
  const toggle = (k: keyof Omit<Layers, "opacity">) => setLayers((l) => ({ ...l, [k]: !l[k] }));
  const setAi = (ai: boolean) => setLayers((l) => (l.ai === ai ? l : { ...l, ai }));
  const meta = overlay.video;
  const clipAvailable = !!api.clipUrl(video.video_id, "x");
  const reportHref = `#/videos/${encodeURIComponent(video.video_id)}/report`;

  return (
    <>
      <PageHeader
        dense
        crumbs={CRUMBS}
        title={video.filename}
        center={<VideoTabs videoId={video.video_id} active="workbench" />}
        actions={
          <>
            <span className="pill outline">{overlay.match_format === "doubles" ? "双打 · 4 人" : "单打 · 2 人"}</span>
            <button
              className={reviewQueue.length ? "primary" : ""}
              disabled={!reviewQueue.length}
              title="跳到当前时间之后下一个待复核的回合、击球或击球类型（快捷键 n，Shift+N 往回）"
              onClick={() => nextReview(1)}
            >
              <Icon name="flag" />
              {reviewQueue.length ? `下一个待复核 · ${reviewQueue.length}` : "没有待复核项"}
            </button>
            <button className="ghost icon-btn" title="快捷键（?）" aria-label="快捷键" onClick={() => setShowKeys(true)}>
              <Icon name="keyboard" />
            </button>
          </>
        }
      />
      <div className="wb">
        <aside className="wb-left">
          <div className="panel">
            <div className="panel-head">
              <h3>分析概览</h3>
            </div>
            <div className="mini-stats">
              <div className="mini-stat">
                <b>{events.rallies.length}</b>
                <span>回合</span>
              </div>
              <div className="mini-stat">
                <b>{events.shots.length}</b>
                <span>击球</span>
              </div>
              <div className={`mini-stat${reviewQueue.length ? " warn" : ""}`} title="还需要人工看一眼的回合、击球和击球类型">
                <b>{reviewQueue.length}</b>
                <span>待复核</span>
              </div>
              <div className="mini-stat" title="已做修正">
                <b>{corrections.length}</b>
                <span>已修正</span>
              </div>
            </div>
            <Stat label="规格">
              {meta.width}×{meta.height} · {Math.round(meta.fps)}fps{meta.variable_frame_rate ? "（可变帧率）" : ""}
            </Stat>
            <Stat label="待复核 回合 / 击球">
              <span className={reviewShots + reviewRallies ? "warn-text" : undefined}>
                {reviewRallies} / {reviewShots}
              </span>
            </Stat>
            <Stat label="击球类型 待复核 / 不可判定">
              <span className={reviewTypes ? "warn-text" : undefined}>
                {events.shots.filter((s) => s.stroke?.source === "model" && s.stroke.status === "needs_review").length} /{" "}
                {events.shots.filter((s) => s.stroke?.source === "model" && s.stroke.status === "undeterminable").length}
              </span>
            </Stat>
            <div className="kv">
              <span className="k">比赛类型</span>
              <span className="v">
                <select
                  className="small"
                  value={formatSetting ?? "auto"}
                  disabled={formatSetting === null}
                  title="自动识别不对时手动指定；会从球员识别开始重新分析"
                  onChange={(e) => void changeFormat(e.target.value as MatchFormatSetting)}
                >
                  <option value="auto">自动识别</option>
                  <option value="singles">强制单打</option>
                  <option value="doubles">强制双打</option>
                </select>
              </span>
            </div>
            <div className="kv">
              <span className="k">球场标定</span>
              <span className="v">
                {overlay.court.length > 1 && <span className="muted">{overlay.court.length} 段</span>}
                {court ? <StatusPill status={court.status} /> : <span className="pill needs_review">未标定</span>}
              </span>
            </div>
            <div className="divider" />
            <div className="wb-tools">
              <button className={`block${calibrating ? " active" : ""}`} onClick={() => setCalibrating((c) => !c)}>
                <Icon name="target" />
                {calibrating ? "取消点选" : "球场线不对？手动点选四角"}
              </button>
              {calibrating && <p className="hint">按 {CORNER_NAMES.join(" → ")} 的顺序点击外框四角；提交后会从标定开始重新分析。</p>}
              {!!court?.other_courts?.length && (
                <>
                  <p className="hint">画面里还有 {court.other_courts.length} 块场地（虚线框），它们的球员和球已被排除。</p>
                  <button
                    className={`block${choosingCourt ? " active" : ""}`}
                    onClick={() => {
                      setCalibrating(false);
                      setChoosingCourt((c) => !c);
                    }}
                  >
                    <Icon name="swap" />
                    {choosingCourt ? "取消选择" : "分析的不是这块场地？换一块"}
                  </button>
                </>
              )}
            </div>
          </div>
          <QualityPanel videoId={video.video_id} />
          <HeatmapPanel overlay={overlay} rally={rally} rallyNo={rally ? rallyNo(rally.rally_id) : null} courtStatus={court?.status} />
          <StrokePanel shots={events.shots} rally={rally} rallyNo={rally ? rallyNo(rally.rally_id) : null} clock={clock} />
        </aside>

        <section className="wb-center">
          <div className="stage-bar">
            <div className="seg" role="group" aria-label="画面" title="原画 / AI 标注切换">
              <button className={layers.ai ? "on" : ""} aria-pressed={layers.ai} onClick={() => setAi(true)}>
                AI 标注
              </button>
              <button className={layers.ai ? "" : "on"} aria-pressed={!layers.ai} onClick={() => setAi(false)}>
                原画
              </button>
            </div>
            <span className="sep" />
            {LAYER_TOGGLES.map(([k, label]) => (
              <button key={k} className={`toggle${layers[k] ? " on" : ""}`} aria-pressed={layers[k]} disabled={!layers.ai} onClick={() => toggle(k)}>
                {label}
              </button>
            ))}
            <button
              className={`toggle${layers.abstract ? " on" : ""}`}
              aria-pressed={layers.abstract}
              disabled={!layers.ai}
              title="隐藏原视频画面，只在深色背景上显示识别元素"
              onClick={() => toggle("abstract")}
            >
              纯抽象
            </button>
            <span className="sep" />
            <label className="range">
              透明度
              <input type="range" min={0.2} max={1} step={0.05} value={layers.opacity} onChange={(e) => setLayers((l) => ({ ...l, opacity: Number(e.target.value) }))} />
              <span className="mono range-value">{Math.round(layers.opacity * 100)}%</span>
            </label>
            <span className="spacer" />
            <div className="legend" title="球：实线=检测，虚线=插值">
              {playerIdsFor(overlay.match_format).map((id) => (
                <span key={id}>
                  <span className="dot" style={{ background: colorOf(id) }} />
                  {labelOf(id)}
                </span>
              ))}
              <span>
                <i className="legend-line" />
                球 · 检测
              </span>
              <span>
                <i className="legend-line dashed" />
                球 · 插值
              </span>
            </div>
          </div>
          <VideoStage overlay={overlay} videoUrl={api.videoFileUrl(video.video_id)} clock={clock} layers={layers} calibrating={calibrating} onCalibrated={onCalibrated} choosingCourt={choosingCourt} onCourtChosen={onCalibrated} />
        </section>

        <aside className="wb-right">
          <MiniMap overlay={overlay} clock={clock} />
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
            <div className="panel-head">
              <Icon name="download" />
              <h3>导出</h3>
            </div>
            <div className="export-list">
              <button className="block" onClick={() => void doExport("json")}>
                <span className="fmt">JSON</span>
                <span className="what">含原始预测和修正</span>
              </button>
              <button className="block" onClick={() => void doExport("csv")}>
                <span className="fmt">CSV</span>
                <span className="what">逐拍</span>
              </button>
              <button
                className="block"
                disabled={!rally || !api.clipUrl(video.video_id, rally.rally_id)}
                title={clipAvailable ? "导出当前回合的视频片段（前后各留 1 秒）" : "演示模式没有原视频"}
                onClick={() => {
                  const url = rally && api.clipUrl(video.video_id, rally.rally_id);
                  if (url) window.location.href = url;
                }}
              >
                <span className="fmt">MP4</span>
                <span className="what">{rally ? `第 ${rallyNo(rally.rally_id)} 回合片段` : "回合片段（先播放到某个回合）"}</span>
              </button>
            </div>
            <div className="divider" />
            <a className="btn block" href={reportHref} title="回合、跑动、覆盖、得失分统计和 AI 教练点评（基于修正后的数据）">
              <Icon name="report" />
              查看赛后报告
            </a>
          </div>
        </aside>

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
            onSelectRally={(id) => {
              const r = events.rallies.find((x) => x.rally_id === id);
              if (r) clock.seek(r.start_ms);
            }}
            onSelectShot={seekShot}
          />
        </footer>
      </div>
      {showKeys && <ShortcutHelp onClose={() => setShowKeys(false)} />}
    </>
  );
}

function Stat({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="kv">
      <span className="k">{label}</span>
      <span className="v mono">{children}</span>
    </div>
  );
}

const SHORTCUTS: [string[], string][] = [
  [["空格"], "播放 / 暂停"],
  [["←", "→"], "上一帧 / 下一帧"],
  [["j", "k"], "上一拍 / 下一拍"],
  [["[", "]"], "上一个 / 下一个回合"],
  [["n", "Shift+N"], "下一个 / 上一个待复核项"],
  [["?"], "显示或关闭这个面板"],
];

function ShortcutHelp({ onClose }: { onClose: () => void }) {
  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="card modal keys-modal" role="dialog" aria-modal="true" aria-label="快捷键" onClick={(e) => e.stopPropagation()}>
        <div className="card-head">
          <Icon name="keyboard" className="lg" />
          <h2>快捷键</h2>
          <span className="spacer" />
          <button className="ghost small" onClick={onClose}>
            关闭
            <kbd>Esc</kbd>
          </button>
        </div>
        <table>
          <tbody>
            {SHORTCUTS.map(([keys, what]) => (
              <tr key={what}>
                <td className="keys-cell">
                  {keys.map((k, i) => (
                    <span key={k}>
                      {i > 0 && <span className="muted"> / </span>}
                      <kbd>{k}</kbd>
                    </span>
                  ))}
                </td>
                <td>{what}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="card-foot">在输入框里打字时快捷键不生效。</p>
      </div>
    </div>
  );
}
