import { useState } from "react";
import { api, type PublicConfig, type User } from "../lib/api";

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
    <main className="page auth-page">
      <form className="card auth-card" onSubmit={submit}>
        <h2>{mode === "login" ? "登录 RallyIQ" : "注册账号"}</h2>
        <label>
          <span className="small muted">邮箱</span>
          <input type="email" autoComplete="email" required value={email} onChange={(e) => setEmail(e.target.value)} />
        </label>
        <label>
          <span className="small muted">密码{mode === "signup" ? "（至少 8 位）" : ""}</span>
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
            <span className="small muted">邀请码</span>
            <input required value={invite} onChange={(e) => setInvite(e.target.value)} />
          </label>
        )}
        {error && <p className="error small">{error}</p>}
        <button type="submit" className="primary" disabled={busy}>
          {busy ? "请稍候…" : mode === "login" ? "登录" : "注册并登录"}
        </button>
        {canSignup && (
          <p className="small muted">
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
  );
}
