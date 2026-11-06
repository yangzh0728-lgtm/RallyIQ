import { useEffect, useRef, useState } from "react";
import type { PlaybackClock } from "../lib/clock";
import { CALIB_ORDER_HINT, drawOverlay, type Layers } from "../lib/overlay";
import { colorOf } from "../lib/players";
import type { OverlayData, Point } from "../lib/types";

interface Props {
  overlay: OverlayData;
  videoUrl: string | null;
  clock: PlaybackClock;
  layers: Layers;
  calibrating: boolean;
  onCalibrated: (corners: Point[]) => void;
}

export function VideoStage({ overlay, videoUrl, clock, layers, calibrating, onCalibrated }: Props) {
  const { width: vw, height: vh } = overlay.video;
  const outer = useRef<HTMLDivElement>(null);
  const video = useRef<HTMLVideoElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const [size, setSize] = useState({ w: 0, h: 0 });
  const [calib, setCalib] = useState<Point[]>([]);
  const [videoError, setVideoError] = useState<string | null>(null);

  // Fit the video's aspect ratio into the available area.
  useEffect(() => {
    const el = outer.current!;
    const ro = new ResizeObserver(() => {
      const { clientWidth: W, clientHeight: H } = el;
      const ar = vw / vh;
      const w = Math.min(W, H * ar);
      setSize({ w: Math.floor(w), h: Math.floor(w / ar) });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, [vw, vh]);

  useEffect(() => {
    clock.attach(video.current);
    return () => clock.detach();
  }, [clock, videoUrl]);

  useEffect(() => setCalib([]), [calibrating]);

  // Redraw on every clock tick (outside React rendering) and when inputs change.
  useEffect(() => {
    const c = canvas.current;
    if (!c || !size.w) return;
    const dpr = window.devicePixelRatio || 1;
    c.width = Math.round(size.w * dpr);
    c.height = Math.round(size.h * dpr);
    const ctx = c.getContext("2d")!;
    const draw = () => drawOverlay(ctx, overlay, clock.ms, layers, colorOf, calibrating ? calib : null);
    draw();
    return clock.subscribe(draw);
  }, [overlay, clock, layers, size, calibrating, calib]);

  const onCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!calibrating) {
      clock.toggle();
      return;
    }
    const r = e.currentTarget.getBoundingClientRect();
    const p: Point = [((e.clientX - r.left) / r.width) * vw, ((e.clientY - r.top) / r.height) * vh];
    const next = [...calib, p];
    setCalib(next);
    if (next.length === 4) onCalibrated(next.map(([x, y]) => [Math.round(x), Math.round(y)]));
  };

  return (
    <div className="stage" ref={outer}>
      <div className={`stage-inner ${calibrating ? "calibrating" : ""}`} style={{ width: size.w, height: size.h }}>
        {videoUrl ? (
          <video
            ref={video}
            src={videoUrl}
            playsInline
            preload="auto"
            onError={() => setVideoError("视频无法播放（格式不支持或文件不可用）；叠加层仍可查看")}
          />
        ) : (
          <div style={{ position: "absolute", inset: 0, background: "radial-gradient(#1b2a1f, #0b130e)" }} />
        )}
        <canvas ref={canvas} onClick={onCanvasClick} />
      </div>
      {calibrating && (
        <div className="stage-hint">
          {CALIB_ORDER_HINT}（{calib.length}/4）
        </div>
      )}
      {!videoUrl && !calibrating && <div className="stage-hint muted">合成数据：无原视频，仅显示 AI 叠加层</div>}
      {videoError && <div className="stage-hint error">{videoError}</div>}
    </div>
  );
}
