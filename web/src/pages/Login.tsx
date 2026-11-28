import { useState } from "react";
import { BrandMark, CourtGlyph, Icon, type IconName } from "../components/ui";
import { api, type PublicConfig, type User } from "../lib/api";

/** What the product does, shown on the brand side of the sign-in screen. */
const POINTS: { icon: IconName; title: string; text: string }[] = [
  { icon: "sparkles", title: "自动识别", text: "球场、球员、羽毛球轨迹和每一拍，上传后自动分析" },
  { icon: "edit", title: "结果可复核", text: "每项结果都带状态和证据，可以逐拍确认或修改" },
  { icon: "trend", title: "报告与档案", text: "赛后报告加长期档案，看清每一场和每一段时间的变化" },
];

export function Login({ config, onDone }: { config: PublicConfig; onDone: (u: User) => void }) {
  const canSignup = config.signup !== "closed";
  const [mode, setMode] = useState<"login" | "signup">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [invite, setInvite] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const u = mode === "login" ? await api.login(email, password) : await api.signup(email, password, invite);
      onDone(u);
    } catch (err) {
      setError((err as Error).message.replace(/^\d+\s*/, ""));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="auth">
      <aside className="auth-brand">
        <div className="brand">
          <BrandMark />
          <span>
            Rally<b>IQ</b>
          </span>
        </div>
        <h1>
          把每一场球，
          <br />
          <em>变成看得懂的数据</em>
        </h1>
        <div className="auth-points">
          {POINTS.map((p) => (
            <div key={p.title}>
              <span className="n">
                <Icon name={p.icon} />
              </span>
              <p>
                <b>{p.title}</b>
                <span>{p.text}</span>
              </p>
            </div>
          ))}
        </div>
        <CourtGlyph className="auth-court" />
      </aside>
      <main className="auth-form">
        <form className="auth-card" onSubmit={submit}>
          <div className="brand auth-brand-compact">
            <BrandMark />
            <span>
              Rally<b>IQ</b>
            </span>
          </div>
          <div className="auth-title">
            <h2>{mode === "login" ? "登录 RallyIQ" : "注册账号"}</h2>
            <p className="muted">{mode === "login" ? "用邮箱和密码登录，继续查看你的比赛分析。" : config.signup === "invite" ? "用邮箱注册，需要一个邀请码。" : "用邮箱注册，马上开始分析比赛。"}</p>
          </div>
          <label>
            <span>邮箱</span>
            <input type="email" autoComplete="email" required placeholder="name@example.com" value={email} onChange={(e) => setEmail(e.target.value)} />
          </label>
          <label>
            <span>密码{mode === "signup" ? "（至少 8 位）" : ""}</span>
            <input
              type="password"
              autoComplete={mode === "login" ? "current-password" : "new-password"}
              required
              minLength={mode === "signup" ? 8 : undefined}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
            />
          </label>
          {mode === "signup" && config.signup === "invite" && (
            <label>
              <span>邀请码</span>
              <input required value={invite} onChange={(e) => setInvite(e.target.value)} />
            </label>
          )}
          {error && (
            <div className="notice bad" role="alert">
              <Icon name="alert" />
              <span>{error}</span>
            </div>
          )}
          <button type="submit" className="primary block" disabled={busy}>
            {busy ? "请稍候…" : mode === "login" ? "登录" : "注册并登录"}
          </button>
          {canSignup && (
            <p className="small muted auth-switch">
              {mode === "login" ? "还没有账号？" : "已有账号？"}{" "}
              <a
                href="#/"
                onClick={(e) => {
                  e.preventDefault();
                  setError(null);
                  setMode(mode === "login" ? "signup" : "login");
                }}
              >
                {mode === "login" ? "注册" : "登录"}
              </a>
            </p>
          )}
        </form>
      </main>
    </div>
  );
}
