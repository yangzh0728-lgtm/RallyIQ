import { useEffect, useState } from "react";
import { BrandMark, Icon, type IconName } from "./components/ui";
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

const NAV_KEY = "rallyiq.nav.collapsed";
function readNavPref(): boolean | null {
  try {
    const v = localStorage.getItem(NAV_KEY);
    return v === null ? null : v === "1";
  } catch {
    return null;
  }
}

function NavItem({ href, icon, label, on }: { href: string; icon: IconName; label: string; on: boolean }) {
  return (
    <a href={href} className={`nav-item${on ? " on" : ""}`} title={label} aria-current={on ? "page" : undefined}>
      <Icon name={icon} />
      <span className="hide-collapsed">{label}</span>
    </a>
  );
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

  // The workbench needs the width, so the sidebar starts collapsed there unless the user chose otherwise.
  const [navPref, setNavPref] = useState<boolean | null>(readNavPref);
  const inWorkbench = !!m && !m[2];
  const collapsed = navPref ?? inWorkbench;
  const toggleNav = () => {
    const next = !collapsed;
    setNavPref(next);
    try {
      localStorage.setItem(NAV_KEY, next ? "1" : "0");
    } catch {
      /* private mode: the choice lasts for this page load only */
    }
  };

  const logout = async () => {
    await api.logout().catch(() => undefined);
    setSession((cur) => (cur.state === "ready" ? { ...cur, user: null } : cur));
    location.hash = "#/";
  };

  if (needLogin && session.state === "ready") {
    return <Login config={session.config} onDone={(u) => setSession({ ...session, user: u })} />;
  }

  const videoBase = m ? `#/videos/${m[1]}` : null;
  return (
    <div className={`app${collapsed ? " nav-collapsed" : ""}`}>
      <aside className="nav">
        <a href="#/" className="brand" title="RallyIQ">
          <BrandMark />
          <span className="hide-collapsed">
            Rally<b>IQ</b>
          </span>
        </a>
        <div className="nav-label">
          <span className="hide-collapsed">工作区</span>
        </div>
        <NavItem href="#/" icon="library" label="比赛库" on={!m && !pm} />
        <NavItem href="#/profile" icon="user" label="球员档案" on={!!pm} />
        {videoBase && (
          <>
            <div className="nav-label">
              <span className="hide-collapsed">当前比赛</span>
            </div>
            <NavItem href={videoBase} icon="workbench" label="回放与纠错" on={!m![2]} />
            <NavItem href={`${videoBase}/report`} icon="report" label="赛后报告" on={!!m![2]} />
          </>
        )}
        <div className="nav-foot">
          {api.mode === "mock" ? (
            <div className="nav-card demo hide-collapsed" title="使用前端合成数据，未连接后端">
              <b>演示模式</b>
              <span className="muted">正在使用合成数据，修正只保存在这个浏览器里。</span>
            </div>
          ) : (
            <a className="nav-item hide-collapsed" href="?mock#/" title="不连接后端，用合成数据看界面">
              <Icon name="sparkles" />
              <span>演示模式</span>
            </a>
          )}
          {user && (
            <div className="nav-user" title={user.email}>
              <span className="avatar">{user.email.slice(0, 1).toUpperCase()}</span>
              <span className="ellipsis small hide-collapsed" style={{ flex: 1 }}>
                {user.email}
              </span>
              <button className="ghost icon-btn small hide-collapsed" title="退出登录" aria-label="退出登录" onClick={() => void logout()}>
                <Icon name="logout" />
              </button>
            </div>
          )}
          <button className="ghost nav-collapse" onClick={toggleNav} title={collapsed ? "展开侧栏" : "收起侧栏"} aria-label={collapsed ? "展开侧栏" : "收起侧栏"}>
            <Icon name="sidebar" />
            <span className="hide-collapsed small">收起侧栏</span>
          </button>
        </div>
      </aside>
      <div className="main">
        {session.state === "loading" ? (
          <div className="page-scroll">
            <main className="page">
              <p className="muted">加载中…</p>
            </main>
          </div>
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
      </div>
    </div>
  );
}
