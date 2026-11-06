import type { ReactNode } from "react";
import { COURT_L, COURT_W, courtLinesM, NET_LINE_M } from "../lib/court";

const PAD = 0.6;

/**
 * Top-down court. Children draw in court metres (y = 0 near baseline); the group is flipped so the
 * near side (camera side) is at the bottom, matching the video.
 */
export function CourtView({ children, height = 300 }: { children?: ReactNode; height?: number }) {
  return (
    <svg
      viewBox={`${-PAD} ${-PAD} ${COURT_W + PAD * 2} ${COURT_L + PAD * 2}`}
      style={{ height, display: "block", margin: "0 auto", background: "#173326", borderRadius: 6 }}
    >
      <g transform={`translate(0 ${COURT_L}) scale(1 -1)`}>
        {children}
        <g stroke="#ffffffaa" strokeWidth={0.04} fill="none">
          {courtLinesM().map(([[x1, y1], [x2, y2]], i) => (
            <line key={i} x1={x1} y1={y1} x2={x2} y2={y2} />
          ))}
        </g>
        <line
          x1={NET_LINE_M[0][0] - 0.3}
          y1={NET_LINE_M[0][1]}
          x2={NET_LINE_M[1][0] + 0.3}
          y2={NET_LINE_M[1][1]}
          stroke="#e5e7eb"
          strokeWidth={0.08}
          strokeDasharray="0.15 0.08"
        />
      </g>
      <text x={-PAD + 0.15} y={COURT_L + PAD - 0.15} fontSize={0.35} fill="#ffffff80">
        近端（机位）
      </text>
    </svg>
  );
}
