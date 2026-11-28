import { useEffect, useState } from "react";
import { api } from "../lib/api";
import { colorOf } from "../lib/players";
import type { PeopleAssignment, PlayerId } from "../lib/types";
import { Icon } from "./ui";

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
    <section className="card memory-card">
      <div className="card-head">
        <Icon name="database" className="head-icon" />
        <h2>写入个人档案</h2>
        <span className="spacer" />
        <a className="btn small" href="#/profile">
          查看我的档案
          <Icon name="arrowRight" />
        </a>
      </div>
      <p className="section-intro">修正和确认完之后再写入：档案只记录你确认过的结果，并按时间对比趋势、复测上次的训练建议；对手会生成习惯报告。</p>
      <div className="people-grid">
        <div className="people-row people-cols" aria-hidden="true">
          <span>球员</span>
          <span>名字</span>
          <span>哪一位是你</span>
        </div>
        {playerIds.map((pid) => (
          <div key={pid} className="people-row">
            <span className="who">
              <span className="dot" style={{ background: colorOf(pid) }} />
              <span className="mono">{pid}</span>
            </span>
            <input
              value={a.people[pid] ?? ""}
              placeholder={pid === a.self_player ? "我" : "名字（留空则不写入）"}
              aria-label={`${pid} 的名字`}
              onChange={(e) => setName(pid, e.target.value)}
            />
            <label className={`self-pick${a.self_player === pid ? " on" : ""}`}>
              <input type="radio" name="self" checked={a.self_player === pid} onChange={() => setA({ ...a, self_player: pid, people: { ...a.people, [pid]: a.people[pid] || "我" } })} /> 是我
            </label>
          </div>
        ))}
      </div>
      <div className="row memory-actions">
        <button className="primary" disabled={busy || named.length === 0} onClick={() => void run("commit")}>
          {!busy && <Icon name="check" />}
          {busy ? "处理中…" : "确认并写入档案"}
        </button>
        <button disabled={busy} onClick={() => void run("withdraw")}>
          从档案撤下
        </button>
        {named.length === 0 && <span className="muted small">至少填写一位球员的名字后才能写入。</span>}
      </div>
      {msg &&
        (msg.ok ? (
          <div className="notice memory-msg">
            <Icon name="check" />
            <span>{msg.text}</span>
          </div>
        ) : (
          <div className="notice bad memory-msg error">
            <Icon name="alert" />
            <span>{msg.text}</span>
          </div>
        ))}
    </section>
  );
}
