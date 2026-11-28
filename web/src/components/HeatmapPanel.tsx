import { useMemo, useState } from "react";
import { computeHeat } from "../lib/heatmap";
import { colorOf, labelOf } from "../lib/players";
import { TEAM_IDS, TEAM_MEMBERS, playerIdsFor, type OverlayData, type PlayerId, type Rally, type Status, type TeamId } from "../lib/types";
import { CourtView } from "./CourtView";

interface Props {
  overlay: OverlayData;
  rally: Rally | undefined;
  rallyNo: number | null;
  courtStatus: Status | undefined;
}

export function HeatmapPanel({ overlay, rally, rallyNo, courtStatus }: Props) {
  const doubles = overlay.match_format === "doubles";
  const ids = playerIdsFor(overlay.match_format);
  // A player id, or a team (doubles: both partners on one map).
  const [sel, setSel] = useState<PlayerId | TeamId>("P1");
  const who: PlayerId[] = sel === "A" || sel === "B" ? TEAM_MEMBERS[sel] : [sel];
  const [scope, setScope] = useState<"rally" | "all">("all");
  const useRally = scope === "rally" && rally;
  const from = useRally ? rally.start_ms : 0;
  const to = useRally ? rally.end_ms : overlay.video.duration_ms;
  const heat = useMemo(() => computeHeat(overlay.frames, who, from, to), [overlay.frames, sel, from, to]); // eslint-disable-line react-hooks/exhaustive-deps
  const color = colorOf(sel);
  // README: only show metres when the spatial calibration is reliable.
  const reliable = courtStatus === "ok";
  const seconds = heat.samples / overlay.video.fps;

  return (
    <div className="panel">
      <div className="panel-head">
        <h3>跑位热力图</h3>
        <span className="spacer" />
        <select className="small" value={sel} aria-label="球员或队伍" onChange={(e) => setSel(e.target.value as PlayerId | TeamId)}>
          {ids.map((p) => (
            <option key={p} value={p}>
              {labelOf(p)}
            </option>
          ))}
          {doubles &&
            TEAM_IDS.map((t) => (
              <option key={t} value={t}>
                {labelOf(t)}
              </option>
            ))}
        </select>
      </div>
      <div className="seg scope" role="group" aria-label="范围">
        <button className={scope === "all" ? "on" : ""} aria-pressed={scope === "all"} onClick={() => setScope("all")}>
          整段视频
        </button>
        <button className={scope === "rally" ? "on" : ""} aria-pressed={scope === "rally"} disabled={!rally} onClick={() => setScope("rally")}>
          当前回合{rallyNo ? ` #${rallyNo}` : ""}
        </button>
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
      <div className="small panel-foot">
        <div className="kv">
          <span className="k">样本</span>
          <span className="v mono">
            {heat.samples} 帧 · {seconds.toFixed(1)} 秒
          </span>
        </div>
        <div className="kv">
          <span className="k">{who.length > 1 ? "两人合计跑动" : "跑动距离"}</span>
          <span className="v">{reliable ? <span className="mono">{heat.distanceM.toFixed(1)} m</span> : <span className="pill needs_review">标定未确认，不显示米</span>}</span>
        </div>
        <div className="muted">{heat.samples < overlay.video.fps * 5 ? "样本太少，仅供参考。" : "基于自动足点估计，未经人工复核。"}</div>
      </div>
    </div>
  );
}
