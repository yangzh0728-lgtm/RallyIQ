// Shared building blocks of the redesigned shell: icons, page frame, headers, tiles, empty states.
import type { CSSProperties, ReactNode } from "react";
import { colorOf } from "../lib/players";

const ICONS = {
  library: "M4 5h7v6H4zM13 5h7v6h-7zM4 13h7v6H4zM13 13h7v6h-7z",
  user: "M12 12a4 4 0 1 0 0-8 4 4 0 0 0 0 8zM4.5 20a7.5 7.5 0 0 1 15 0",
  play: "M8 5.5v13l10.5-6.5z",
  pause: "M8 5v14M16 5v14",
  stepBack: "M17 6v12L8.5 12zM6.5 6v12",
  stepForward: "M7 6v12l8.5-6zM17.5 6v12",
  upload: "M12 16V4M7 8.5 12 4l5 4.5M4.5 15.5v3A1.5 1.5 0 0 0 6 20h12a1.5 1.5 0 0 0 1.5-1.5v-3",
  download: "M12 4v12M7 11.5l5 4.5 5-4.5M4.5 15.5v3A1.5 1.5 0 0 0 6 20h12a1.5 1.5 0 0 0 1.5-1.5v-3",
  report: "M5 20V10M12 20V4M19 20v-7M3 20h18",
  workbench: "M3.5 5.5h17v10h-17zM9 19.5h6M12 15.5v4M10 8.5v4l3.5-2z",
  trash: "M4.5 7h15M9.5 7V4.5h5V7M6.5 7l.8 12.5h9.4L17.5 7M10 10.5v6M14 10.5v6",
  chevronLeft: "M14.5 5.5 8 12l6.5 6.5",
  chevronRight: "M9.5 5.5 16 12l-6.5 6.5",
  chevronDown: "M5.5 9.5 12 16l6.5-6.5",
  arrowRight: "M4.5 12h15M13.5 6l6 6-6 6",
  keyboard: "M3 6.5h18v11H3zM7 10.5h.01M11 10.5h.01M15 10.5h.01M7.5 14h9",
  check: "M5 12.5 10 17.5 19.5 7",
  alert: "M12 4 2.8 19.5h18.4zM12 10v4.5M12 17.2v.1",
  sparkles: "M11 4l1.6 4.4L17 10l-4.4 1.6L11 16l-1.6-4.4L5 10l4.4-1.6zM18 15l.8 2.2L21 18l-2.2.8L18 21l-.8-2.2L15 18l2.2-.8z",
  target: "M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18zM12 16a4 4 0 1 0 0-8 4 4 0 0 0 0 8zM12 12h.01",
  search: "M11 18a7 7 0 1 0 0-14 7 7 0 0 0 0 14zM20 20l-4-4",
  flag: "M5.5 21V4M5.5 5h11l-2 4 2 4h-11",
  clock: "M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18zM12 7v5.2l3.3 2",
  layers: "M12 4 3 9l9 5 9-5zM3 14.5l9 5 9-5",
  logout: "M10 4H5.5v16H10M15 8l4 4-4 4M9 12h10",
  refresh: "M19.5 12a7.5 7.5 0 1 1-2.4-5.5M19.5 4.5v4h-4",
  video: "M3.5 6.5h12v11h-12zM15.5 10.5l5-3v9l-5-3",
  trend: "M3.5 17.5 9 12l4 4 7.5-8.5M15 7.5h5.5V13",
  zoom: "M10.5 17a6.5 6.5 0 1 0 0-13 6.5 6.5 0 0 0 0 13zM20 20l-4.8-4.8M10.5 8v5M8 10.5h5",
  court: "M5 3.5h14v17H5zM5 12h14M12 3.5v5.5M12 15v5.5M5 9h14M5 15h14",
  edit: "M4.5 19.5h4L19 9l-4-4L4.5 15.5zM13.5 6.5l4 4",
  sidebar: "M3.5 5h17v14h-17zM9 5v14",
  info: "M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18zM12 11v5.5M12 7.6v.1",
  swap: "M4 8.5h14l-3.5-3.5M20 15.5H6l3.5 3.5",
  heat: "M12 21a6 6 0 0 0 6-6c0-3.5-2.5-5-3.5-8-2 1-3 3-3 4.5C10 9.5 9.5 7 8 6c-1 2.5-2 4.5-2 9a6 6 0 0 0 6 6z",
  database: "M4.5 6.5c0-1.4 3.4-2.5 7.5-2.5s7.5 1.1 7.5 2.5-3.4 2.5-7.5 2.5-7.5-1.1-7.5-2.5zM4.5 6.5v11c0 1.4 3.4 2.5 7.5 2.5s7.5-1.1 7.5-2.5v-11M4.5 12c0 1.4 3.4 2.5 7.5 2.5s7.5-1.1 7.5-2.5",
} as const;

export type IconName = keyof typeof ICONS;

const FILLED: Partial<Record<IconName, true>> = { play: true, stepBack: true, stepForward: true };

export function Icon({ name, className = "", style }: { name: IconName; className?: string; style?: CSSProperties }) {
  const fill = FILLED[name];
  return (
    <svg
      className={`icon ${className}`}
      style={style}
      viewBox="0 0 24 24"
      fill={fill ? "currentColor" : "none"}
      stroke="currentColor"
      strokeWidth={fill ? 1.2 : 1.7}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d={ICONS[name]} />
    </svg>
  );
}

/** Brand mark: a shuttle in flight, drawn from a cork and three feathers. */
export function BrandMark() {
  return (
    <span className="brand-mark" aria-hidden="true">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.8} strokeLinecap="round" strokeLinejoin="round">
        <path d="M8.6 12.9 11.6 3.6C15.4 4.4 19.6 8.6 20.4 12.4L11.1 15.4M9.8 14.2 16.8 7.2" />
        <circle cx="7.2" cy="16.8" r="3.4" fill="currentColor" stroke="none" />
      </svg>
    </span>
  );
}

export interface Crumb {
  label: string;
  href?: string;
}

interface HeaderProps {
  title: ReactNode;
  crumbs?: Crumb[];
  /** Shown between the title and the actions (e.g. the workbench / report switch). */
  center?: ReactNode;
  actions?: ReactNode;
  dense?: boolean;
}

export function PageHeader({ title, crumbs, center, actions, dense }: HeaderProps) {
  return (
    <header className={`page-header${dense ? " dense" : ""}`}>
      <div className="titles">
        {crumbs && crumbs.length > 0 && (
          <nav className="crumbs" aria-label="位置">
            {crumbs.map((c, i) => (
              <span key={i} className="row nowrap" style={{ gap: 6, minWidth: 0 }}>
                {i > 0 && <span className="sep">/</span>}
                {c.href ? <a href={c.href}>{c.label}</a> : <span className="ellipsis">{c.label}</span>}
              </span>
            ))}
          </nav>
        )}
        <h1 title={typeof title === "string" ? title : undefined}>{title}</h1>
      </div>
      {center}
      {actions && <div className="actions">{actions}</div>}
    </header>
  );
}

/** Standard scrolling page: sticky header (and optional section tabs) over a centred column. */
export function Page({ header, subnav, className = "", children }: { header: ReactNode; subnav?: ReactNode; className?: string; children: ReactNode }) {
  return (
    <>
      {header}
      {subnav}
      <div className="page-scroll">
        <main className={`page ${className}`}>{children}</main>
      </div>
    </>
  );
}

/** The two views of one video: playback + corrections, and the post-match report. */
export function VideoTabs({ videoId, active }: { videoId: string; active: "workbench" | "report" }) {
  const base = `#/videos/${encodeURIComponent(videoId)}`;
  return (
    <nav className="seg lg" aria-label="视图">
      <a href={base} className={active === "workbench" ? "on" : ""}>
        <Icon name="workbench" />
        回放与纠错
      </a>
      <a href={`${base}/report`} className={active === "report" ? "on" : ""}>
        <Icon name="report" />
        赛后报告
      </a>
    </nav>
  );
}

export function StatTile({ label, value, unit, sub, icon }: { label: string; value: ReactNode; unit?: string; sub?: ReactNode; icon?: IconName }) {
  return (
    <div className="stat-tile">
      <div className="label">
        {icon && <Icon name={icon} />}
        {label}
      </div>
      <div className="value mono">
        {value}
        {unit && <small>{unit}</small>}
      </div>
      {sub && <div className="sub">{sub}</div>}
    </div>
  );
}

export function Empty({ icon, title, children }: { icon: IconName; title: string; children?: ReactNode }) {
  return (
    <div className="empty">
      <span className="glyph">
        <Icon name={icon} className="lg" />
      </span>
      <b>{title}</b>
      {children && <p className="small">{children}</p>}
    </div>
  );
}

/** Coloured square with the player / team id, used wherever a side needs a face. */
export function SideBadge({ id, label }: { id: string; label?: string }) {
  return (
    <span className="sb-badge" style={{ background: colorOf(id) }}>
      {label ?? id}
    </span>
  );
}

/** Small top-down court drawing for thumbnails and decoration. */
export function CourtGlyph({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 46 34" fill="none" stroke="currentColor" strokeWidth={1} aria-hidden="true">
      <rect x="3" y="3" width="40" height="28" rx="1" />
      <path d="M23 3v28M3 7h40M3 27h40M6 3v28M40 3v28M15 7v20M31 7v20M15 17h-9M31 17h9" />
    </svg>
  );
}
