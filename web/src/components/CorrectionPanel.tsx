import type { PlaybackClock } from "../lib/clock";
import { describeCorrection, formatMs } from "../lib/corrections";
import { colorOf, labelOf } from "../lib/players";
import { useClockMs } from "../lib/useClock";
import { PLAYER_IDS, type Correction, type CorrectionOp, type PlayerId, type Rally, type ShotEvent, type Source, type Status } from "../lib/types";

const STATUS_LABEL: Record<Status, string> = { ok: "可用", needs_review: "待复核", undeterminable: "不可判定" };

export function StatusPill({ status, source }: { status: Status; source?: Source }) {
  return source === "human" ? <span className="pill human">人工</span> : <span className={`pill ${status}`}>{STATUS_LABEL[status] ?? status}</span>;
}

interface Props {
  rally: Rally | undefined;
  rallyNo: number | null;
  shots: ShotEvent[];
  selectedShotId: string | null;
  onSelectShot: (id: string) => void;
  clock: PlaybackClock;
  fps: number;
  submit: (op: CorrectionOp) => void;
  busy: boolean;
  hardExample: boolean;
  setHardExample: (v: boolean) => void;
  corrections: Correction[];
  error: string | null;
}

export function CorrectionPanel(p: Props) {
  const t = useClockMs(p.clock);
  const frame = Math.round((t * p.fps) / 1000);
  const shot = p.shots.find((s) => s.shot_id === p.selectedShotId);
  const r = p.rally;
  const ask = (msg: string) => window.confirm(msg);

  return (
    <>
      <div className="panel">
        <div className="row">
          <h3 style={{ margin: 0 }}>{r ? `回合 ${p.rallyNo}` : "当前不在回合内"}</h3>
          {r && <StatusPill status={r.status} source={r.source} />}
          <span className="spacer" />
          {r && (
            <span className="mono small muted">
              {formatMs(r.start_ms)}–{formatMs(r.end_ms)}
            </span>
          )}
        </div>
        {r && (
          <div className="row small" style={{ margin: "6px 0 8px" }}>
            <span className="muted">得分方</span>
            <select
              disabled={p.busy}
              value={r.winner ?? ""}
              onChange={(e) => p.submit({ op: "set_rally_winner", payload: { rally_id: r.rally_id, winner: (e.target.value || null) as PlayerId | null } })}
            >
              <option value="">待确认</option>
              {PLAYER_IDS.map((id) => (
                <option key={id} value={id}>
                  {labelOf(id)}
                </option>
              ))}
            </select>
            <span className="muted">
              · {p.shots.length} 拍 · 置信度 {Math.round(r.confidence * 100)}%
            </span>
          </div>
        )}
        <div className="row small">
          <button disabled={!r || p.busy || frame >= (r?.end_frame ?? 0)} title="把回合起点设为当前帧" onClick={() => r && p.submit({ op: "set_rally_bounds", payload: { rally_id: r.rally_id, start_frame: frame } })}>
            起点=当前
          </button>
          <button disabled={!r || p.busy || frame <= (r?.start_frame ?? 0)} title="把回合终点设为当前帧" onClick={() => r && p.submit({ op: "set_rally_bounds", payload: { rally_id: r.rally_id, end_frame: frame } })}>
            终点=当前
          </button>
          <button disabled={!r || p.busy || r.source === "human"} onClick={() => r && p.submit({ op: "confirm_rally", payload: { rally_id: r.rally_id } })}>
            确认
          </button>
          <button className="danger" disabled={!r || p.busy} onClick={() => r && ask(`删除回合 ${p.rallyNo}？原始预测会保留。`) && p.submit({ op: "delete_rally", payload: { rally_id: r.rally_id } })}>
            删除
          </button>
          <button
            disabled={p.busy || !!r}
            title={r ? "当前已在一个回合内" : "从当前帧开始新建 5 秒回合，再用起点/终点调整"}
            onClick={() => p.submit({ op: "add_rally", payload: { start_frame: frame, end_frame: frame + Math.round(5 * p.fps) } })}
          >
            新建回合
          </button>
        </div>
      </div>

      <div className="panel">
        <div className="row" style={{ marginBottom: 6 }}>
          <h3 style={{ margin: 0 }}>击球</h3>
          <span className="spacer" />
          <select disabled={p.busy} value="" onChange={(e) => e.target.value && p.submit({ op: "add_shot", payload: { frame, hitter: e.target.value as PlayerId } })}>
            <option value="">＋ 在当前帧加一拍（漏检）</option>
            {PLAYER_IDS.map((id) => (
              <option key={id} value={id}>
                击球者 {labelOf(id)}
              </option>
            ))}
          </select>
        </div>
        <div className="shot-list">
          {p.shots.length === 0 && <div className="muted small">{r ? "这个回合没有击球。" : "播放到某个回合内查看击球。"}</div>}
          {p.shots.map((s, i) => (
            <div key={s.shot_id} className={`shot-item ${s.shot_id === p.selectedShotId ? "selected" : ""}`} onClick={() => p.onSelectShot(s.shot_id)}>
              <span className="muted mono small">{i + 1}</span>
              <span className="mono small">{formatMs(s.t_ms)}</span>
              <span className="row small" style={{ gap: 6 }}>
                <span className="dot" style={{ background: colorOf(s.hitter) }} />
                {labelOf(s.hitter)}
              </span>
              <span className="row" style={{ gap: 4 }}>
                {s.source === "model" && <span className="muted small mono">{Math.round(s.confidence * 100)}%</span>}
                <StatusPill status={s.status} source={s.source} />
              </span>
            </div>
          ))}
        </div>
        {shot && (
          <div style={{ borderTop: "1px solid var(--line)", marginTop: 8, paddingTop: 8, display: "grid", gap: 6 }}>
            <div className="small muted">
              选中 {shot.shot_id} · {formatMs(shot.t_ms)} · 第 {shot.frame} 帧
              {shot.reasons?.length ? ` · 依据：${shot.reasons.join("；")}` : ""}
            </div>
            <div className="row small">
              击球者
              {PLAYER_IDS.map((id) => (
                <button key={id} className={shot.hitter === id ? "active" : ""} disabled={p.busy || shot.hitter === id} onClick={() => p.submit({ op: "set_hitter", payload: { shot_id: shot.shot_id, hitter: id } })}>
                  {labelOf(id)}
                </button>
              ))}
            </div>
            <div className="row small">
              <button disabled={p.busy || frame === shot.frame} onClick={() => p.submit({ op: "move_shot", payload: { shot_id: shot.shot_id, frame } })}>
                移到当前帧
              </button>
              <button disabled={p.busy || shot.source === "human"} onClick={() => p.submit({ op: "confirm_shot", payload: { shot_id: shot.shot_id } })}>
                确认正确
              </button>
              <button className="danger" disabled={p.busy} onClick={() => p.submit({ op: "delete_shot", payload: { shot_id: shot.shot_id } })}>
                删除（误检）
              </button>
            </div>
          </div>
        )}
        <div className="row small" style={{ marginTop: 8 }}>
          <label className="row muted" style={{ gap: 4 }}>
            <input type="checkbox" checked={p.hardExample} onChange={(e) => p.setHardExample(e.target.checked)} />
            修正记为难例（进入评估集）
          </label>
          <span className="spacer" />
          <button
            disabled={p.busy}
            title="球员 ID 跟错时使用：从当前帧起 P1/P2 互换"
            onClick={() => ask(`从 ${formatMs(t)} 起交换 P1/P2 的击球者和得分方？`) && p.submit({ op: "swap_player_ids", payload: { from_frame: frame } })}
          >
            从此帧起交换 P1/P2
          </button>
        </div>
        {p.error && <div className="error small">{p.error}</div>}
      </div>

      <div className="panel">
        <h3>修正记录 · {p.corrections.length}</h3>
        <div className="history">
          {p.corrections.length === 0 && <div className="muted">还没有修正。原始预测始终保留。</div>}
          {[...p.corrections].reverse().map((c) => (
            <div key={c.correction_id} className="row" style={{ gap: 6 }}>
              <span className="muted mono">#{c.seq}</span>
              <span>{describeCorrection(c, p.fps)}</span>
              {c.hard_example && <span className="pill needs_review">难例</span>}
            </div>
          ))}
        </div>
      </div>
    </>
  );
}
