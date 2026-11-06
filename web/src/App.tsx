import { useEffect, useState } from "react";
import { api, UNAUTHORIZED_EVENT, type PublicConfig, type User } from "./lib/api";
import { Home } from "./pages/Home";
import { Login } from "./pages/Login";
import { Profile } from "./pages/Profile";
import { Report } from "./pages/Report";
import { Workbench } from "./pages/Workbench";

function useHashRoute(): string {
  const [hash, setHash] = useState(() => location.hash.slice(1) || "/");
  useEffect(() => {
    const on = () => setHash(location.hash.slice(1) || "/");
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  return hash;
}

type Session = { state: "loading" } | { state: "ready"; config: PublicConfig; user: User | null } | { state: "offline" };

function useSession() {
  const [s, setS] = useState<Session>({ state: "loading" });
  useEffect(() => {
    let alive = true;
    api
      .getConfig()
      .then(async (config) => ({ config, user: config.auth ? await api.me() : null }))
      .then(
        ({ config, user }) => alive && setS({ state: "ready", config, user }),
        // Backend unreachable: keep the old behaviour (pages show their own errors / demo link).
        () => alive && setS({ state: "offline" }),
      );
    const onUnauthorized = () => setS((cur) => (cur.state === "ready" ? { ...cur, user: null } : cur));
    window.addEventListener(UNAUTHORIZED_EVENT, onUnauthorized);
    return () => {
      alive = false;
      window.removeEventListener(UNAUTHORIZED_EVENT, onUnauthorized);
    };
  }, []);
  return [s, setS] as const;
}

export function App() {
  const route = useHashRoute();
  const [session, setSession] = useSession();
  const [path, query = ""] = route.split("?");
  const m = path.match(/^\/videos\/([^/]+)(\/report)?/);
  const pm = path.match(/^\/profile(?:\/(.+))?$/);
  const startMs = Number(new URLSearchParams(query).get("t")) || null;
  const needLogin = session.state === "ready" && session.config.auth && !session.user;
  const user = session.state === "ready" ? session.user : null;

  const logout = async () => {
    await api.logout().catch(() => undefined);
    setSession((cur) => (cur.state === "ready" ? { ...cur, user: null } : cur));
    location.hash = "#/";
  };

  return (
    <>
      <header className="topbar">
        <a href="#/" className="brand">
          Rally<b>IQ</b>
        </a>
        <span className="muted small">单打 / 双打分析工作台</span>
        <span className="spacer" />
        <a className="small" href="#/profile">我的档案</a>
        {api.mode === "mock" ? (
          <span className="mode-badge" title="使用前端合成数据，未连接后端">演示模式 · 合成数据</span>
        ) : (
          <a className="small muted" href="?mock#/">
            切换到演示模式
          </a>
        )}
        {user && (
          <>
            <span className="small muted">{user.email}</span>
            <button className="small" onClick={() => void logout()}>
              退出
            </button>
          </>
        )}
      </header>
      {session.state === "loading" ? (
        <main className="page">
          <p className="muted">加载中…</p>
        </main>
      ) : needLogin && session.state === "ready" ? (
        <Login config={session.config} onDone={(u) => setSession({ ...session, user: u })} />
      ) : m ? (
        m[2] ? (
          <Report key={m[1]} videoId={decodeURIComponent(m[1])} />
        ) : (
          <Workbench key={m[1]} videoId={decodeURIComponent(m[1])} startMs={startMs} />
        )
      ) : pm ? (
        <Profile key={pm[1] ?? ""} person={pm[1] ? decodeURIComponent(pm[1]) : null} />
      ) : (
        <Home />
      )}
    </>
  );
}
