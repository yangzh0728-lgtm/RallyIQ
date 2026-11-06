import { useEffect, useState } from "react";
import { api } from "./lib/api";
import { Home } from "./pages/Home";
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

export function App() {
  const route = useHashRoute();
  const m = route.match(/^\/videos\/([^/]+)(\/report)?/);
  return (
    <>
      <header className="topbar">
        <a href="#/" className="brand">
          Rally<b>IQ</b>
        </a>
        <span className="muted small">单打 / 双打分析工作台</span>
        <span className="spacer" />
        {api.mode === "mock" ? (
          <span className="mode-badge" title="使用前端合成数据，未连接后端">演示模式 · 合成数据</span>
        ) : (
          <a className="small muted" href="?mock#/">
            切换到演示模式
          </a>
        )}
      </header>
      {m ? (
        m[2] ? (
          <Report key={m[1]} videoId={decodeURIComponent(m[1])} />
        ) : (
          <Workbench key={m[1]} videoId={decodeURIComponent(m[1])} />
        )
      ) : (
        <Home />
      )}
    </>
  );
}
