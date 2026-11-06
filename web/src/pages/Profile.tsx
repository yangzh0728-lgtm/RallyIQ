import { useEffect, useState } from "react";
import { api } from "../lib/api";
import { formatMs } from "../lib/corrections";
import type { Habit, OpponentReport, PersonSummary, PlayerProfile, ProfileMetric, RetestResult } from "../lib/types";

const TREND: Record<ProfileMetric["trend"], string> = { up: "↑", down: "↓", flat: "→", new: "新", insufficient: "—" };
const RETEST: Record<RetestResult["status"], string> = {
  improved: "改善", worse: "变差", changed: "有变化", unchanged: "持平", pending: "等下一场", not_measurable: "无法测量",
};
const CONF: Record<string, string> = { high: "高", medium: "中", low: "低" };
const fmt = (v: number | null, unit?: string | null) => (v == null ? "—" : `${Math.round(v * 100) / 100}${unit ?? ""}`);
const link = (videoId: string, tMs: number) => `#/videos/${encodeURIComponent(videoId)}?t=${tMs}`;

export function Profile({ person }: { person: string | null }) {
  const [people, setPeople] = useState<PersonSummary[] | null>(null);
  const [profile, setProfile] = useState<PlayerProfile | null>(null);
  const [opp, setOpp] = useState<OpponentReport | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.listPeople().then(setPeople, () => setPeople([]));
    setProfile(null);
    setOpp(null);
    setError(null);
    api.getProfile(person).then(setProfile, (e: Error) => setError(e.message));
    if (person) api.getOpponent(person).then(setOpp, () => setOpp(null));
  }, [person]);

  return (
    <main className="page report">
      <div className="row">
        <a href="#/">← 视频</a>
        <h1 className="report-title">{person ? `${person} 的档案` : "我的档案"}</h1>
        <span className="spacer" />
        {people?.map((p) => (
          <a key={p.person} href={p.is_self ? "#/profile" : `#/profile/${encodeURIComponent(p.person)}`}>
            <span className={`pill ${(person ?? null) === (p.is_self ? null : p.person) ? "human" : ""}`}>
              {p.person} · {p.videos} 场
            </span>
          </a>
        ))}
      </div>
      {error ? (
        <section className="card">{error}</section>
      ) : !profile ? (
        <section className="card muted">加载中…</section>
      ) : profile.videos.length === 0 ? (
        <section className="card muted">
          还没有写入档案的比赛。在赛后报告页底部“写入个人档案”里填好谁是你、对手是谁，确认后写入。
        </section>
      ) : (
        <>
          {profile.changes.length > 0 && (
            <section className="card">
              <h2>最近变化</h2>
              <ul className="caveats">
                {profile.changes.map((c) => (
                  <li key={c}>{c}</li>
                ))}
              </ul>
            </section>
          )}
          {opp && <OpponentView r={opp} />}
          {profile.retests.length > 0 && (
            <section className="card">
              <h2>训练复测</h2>
              <table>
                <thead>
                  <tr><th>训练</th><th>复测指标</th><th>上次</th><th>这次</th><th>结果</th></tr>
                </thead>
                <tbody>
                  {profile.retests.map((r, i) => (
                    <tr key={i}>
                      <td title={r.why}>{r.drill}</td>
                      <td className="small">{r.label ?? r.metric}</td>
                      <td className="mono">{fmt(r.before, r.unit)}</td>
                      <td className="mono">{fmt(r.after, r.unit)}</td>
                      <td>
                        {RETEST[r.status]}
                        {r.same_conditions === false && <span className="muted small">（拍摄条件不同，仅供参考）</span>}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </section>
          )}
          <section className="card">
            <h2>指标走势（{profile.videos.length} 场）</h2>
            <table>
              <thead>
                <tr><th>指标</th>{profile.videos.map((v) => <th key={v.video_id} title={v.filename ?? v.video_id}>{new Date(v.recorded_at).toLocaleDateString()}</th>)}<th>趋势</th><th>可信度</th></tr>
              </thead>
              <tbody>
                {profile.metrics.map((m) => (
                  <tr key={m.key} title={m.note ?? undefined}>
                    <td>{m.label}</td>
                    {profile.videos.map((v) => {
                      const p = m.points.find((x) => x.video_id === v.video_id);
                      return <td key={v.video_id} className={`mono ${p?.status !== "ok" ? "muted" : ""}`}>{p ? fmt(p.value, m.unit) : ""}</td>;
                    })}
                    <td>{TREND[m.trend]}{m.note && <span className="warn-text"> !</span>}</td>
                    <td className="small">{CONF[m.confidence]}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>
          <section className="card">
            <h2>比赛</h2>
            <table>
              <tbody>
                {profile.videos.map((v) => (
                  <tr key={v.video_id}>
                    <td><a href={`#/videos/${encodeURIComponent(v.video_id)}/report`}>{v.filename ?? v.video_id}</a></td>
                    <td className="muted small">{new Date(v.recorded_at).toLocaleString()}</td>
                    <td className="small">{v.player_id} · 对手 {v.opponents.join("、") || "—"}</td>
                    <td className="muted small">{Object.values(v.conditions).join(" · ")}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="muted small">档案日志共 {profile.log_events} 条记录（只追加，撤下和重新写入都会留痕）。</p>
          </section>
        </>
      )}
    </main>
  );
}

function HabitRow({ h }: { h: Habit }) {
  return (
    <tr>
      <td>{h.condition}</td>
      <td>{h.outcome}</td>
      <td className="mono">{Math.round(h.p * 100)}%（{h.hits}/{h.n}）</td>
      <td className="small">{CONF[h.confidence]}</td>
      <td className="small">
        {h.evidence.map((e) => <a key={e.video_id + e.shot_id} href={link(e.video_id, e.t_ms)}>{formatMs(e.t_ms)} </a>)}
        {h.counter_examples.length > 0 && <span className="muted"> 反例 </span>}
        {h.counter_examples.map((e) => <a key={e.video_id + e.shot_id} className="muted" href={link(e.video_id, e.t_ms)}>{formatMs(e.t_ms)} </a>)}
      </td>
    </tr>
  );
}

function OpponentView({ r }: { r: OpponentReport }) {
  const rows = [...r.habits, ...r.next_shot];
  return (
    <section className="card">
      <h2>对手习惯（{r.videos.length} 场 · {r.shots} 拍，{r.typed_shots} 拍有类型）</h2>
      {r.caveats.length > 0 && (
        <ul className="caveats">
          {r.caveats.map((c) => <li key={c}>{c}</li>)}
        </ul>
      )}
      {rows.length === 0 ? (
        <p className="muted">证据不足：还没有出现至少 5 次、占比 40% 以上的倾向。</p>
      ) : (
        <table>
          <thead>
            <tr><th>情境</th><th>倾向</th><th>占比</th><th>可信度</th><th>视频证据 / 反例</th></tr>
          </thead>
          <tbody>
            {rows.map((h) => <HabitRow key={h.condition + h.outcome} h={h} />)}
          </tbody>
        </table>
      )}
    </section>
  );
}
