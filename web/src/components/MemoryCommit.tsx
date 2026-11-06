import { useEffect, useState } from "react";
import { api } from "../lib/api";
import { colorOf } from "../lib/players";
import type { PeopleAssignment, PlayerId } from "../lib/types";

/** Report page card: name the players, then write this reviewed match into the long-term profile. */
export function MemoryCommit({ videoId, playerIds }: { videoId: string; playerIds: PlayerId[] }) {
  const [a, setA] = useState<PeopleAssignment | null>(null);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<{ ok: boolean; text: string } | null>(null);
  useEffect(() => {
    api.getPeople(videoId).then(setA, (e: Error) => setMsg({ ok: false, text: e.message }));
  }, [videoId]);
  if (!a) return null;

  const setName = (pid: PlayerId, name: string) => setA({ ...a, people: { ...a.people, [pid]: name } });
  const named = playerIds.filter((p) => (a.people[p] ?? "").trim());
  const run = async (what: "commit" | "withdraw") => {
    setBusy(true);
    setMsg(null);
    try {
      if (what === "commit") {
        const saved = await api.setPeople(videoId, a.people, a.self_player && named.includes(a.self_player) ? a.self_player : null);
        setA(saved);
        const ev = await api.commitMemory(videoId);
        setMsg({ ok: true, text: `已写入档案：${ev.length} 名球员。原有记录不会被覆盖，重新写入会追加新版本。` });
      } else {
        await api.withdrawMemory(videoId);
        setMsg({ ok: true, text: "已从档案中撤下这场比赛（日志里保留撤下记录）。" });
      }
    } catch (e) {
      setMsg({ ok: false, text: (e as Error).message });
    } finally {
      setBusy(false);
    }
  };

  return (
    <section className="card">
      <div className="row" style={{ marginBottom: 8 }}>
        <h2 style={{ margin: 0 }}>写入个人档案</h2>
        <span className="spacer" />
        <a href="#/profile">查看我的档案 →</a>
      </div>
      <p className="muted small">修正和确认完之后再写入：档案只记录你确认过的结果，并按时间对比趋势、复测上次的训练建议；对手会生成习惯报告。</p>
      <div className="people-grid">
        {playerIds.map((pid) => (
          <div key={pid} className="row">
            <span className="dot" style={{ background: colorOf(pid) }} />
            <span className="mono">{pid}</span>
            <input
              value={a.people[pid] ?? ""}
              placeholder={pid === a.self_player ? "我" : "名字（留空则不写入）"}
              onChange={(e) => setName(pid, e.target.value)}
            />
            <label className="small">
              <input type="radio" name="self" checked={a.self_player === pid} onChange={() => setA({ ...a, self_player: pid, people: { ...a.people, [pid]: a.people[pid] || "我" } })} /> 是我
            </label>
          </div>
        ))}
      </div>
      <div className="row" style={{ marginTop: 8 }}>
        <button className="primary" disabled={busy || named.length === 0} onClick={() => void run("commit")}>
          {busy ? "处理中…" : "确认并写入档案"}
        </button>
        <button disabled={busy} onClick={() => void run("withdraw")}>
          从档案撤下
        </button>
      </div>
      {msg && <div className={msg.ok ? "notice" : "error"} style={{ marginTop: 8 }}>{msg.text}</div>}
    </section>
  );
}
