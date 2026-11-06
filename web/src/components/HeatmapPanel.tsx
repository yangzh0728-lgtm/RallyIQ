import { useMemo, useState } from "react";
import { computeHeat } from "../lib/heatmap";
import { colorOf, labelOf } from "../lib/players";
import { PLAYER_IDS, type OverlayData, type PlayerId, type Rally, type Status } from "../lib/types";
import { CourtView } from "./CourtView";

interface Props {
  overlay: OverlayData;
  rally: Rally | undefined;
  rallyNo: number | null;
  courtStatus: Status | undefined;
}

export function HeatmapPanel({ overlay, rally, rallyNo, courtStatus }: Props) {
  const [playerId, setPlayerId] = useState<PlayerId>("P1");
  const [scope, setScope] = useState<"rally" | "all">("all");
  const useRally = scope === "rally" && rally;
  const from = useRally ? rally.start_ms : 0;
  const to = useRally ? rally.end_ms : overlay.video.duration_ms;
  const heat = useMemo(() => computeHeat(overlay.frames, playerId, from, to), [overlay.frames, playerId, from, to]);
  const color = colorOf(playerId);
  // README: only show metres when the spatial calibration is reliable.
  const reliable = courtStatus === "ok";
  const seconds = heat.samples / overlay.video.fps;

  return (
    <div className="panel">
      <h3>跑位热力图</h3>
      <div className="row" style={{ marginBottom: 8 }}>
        <select value={playerId} onChange={(e) => setPlayerId(e.target.value as PlayerId)}>
          {PLAYER_IDS.map((p) => (
            <option key={p} value={p}>
              {labelOf(p)}
            </option>
          ))}
        </select>
        <select value={scope} onChange={(e) => setScope(e.target.value as "rally" | "all")}>
          <option value="all">整段视频</option>
          <option value="rally" disabled={!rally}>
            当前回合{rallyNo ? ` #${rallyNo}` : ""}
          </option>
        </select>
      </div>
      <CourtView height={280}>
        {heat.counts.map((row, r) =>
          row.map((n, c) =>
            n ? (
              <rect
                key={`${r}-${c}`}
                x={c * heat.cell}
                y={r * heat.cell}
                width={heat.cell}
                height={heat.cell}
                fill={color}
                opacity={0.15 + 0.8 * Math.sqrt(n / heat.max)}
              />
            ) : null,
          ),
        )}
      </CourtView>
      <div className="small" style={{ display: "grid", gap: 2, marginTop: 8 }}>
        <div className="row">
          <span className="muted">样本</span>
          <span className="spacer" />
          <span className="mono">
            {heat.samples} 帧 · {seconds.toFixed(1)} 秒
          </span>
        </div>
        <div className="row">
          <span className="muted">跑动距离</span>
          <span className="spacer" />
          {reliable ? <span className="mono">{heat.distanceM.toFixed(1)} m</span> : <span className="pill needs_review">标定未确认，不显示米</span>}
        </div>
        <div className="muted">{heat.samples < overlay.video.fps * 5 ? "样本太少，仅供参考。" : "基于自动足点估计，未经人工复核。"}</div>
      </div>
    </div>
  );
}
