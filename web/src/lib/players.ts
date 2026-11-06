import { PLAYER_COLORS, PLAYER_LABELS, type PlayerId } from "./types";

const UNKNOWN = "#64748b";
export const colorOf = (id: PlayerId | null | undefined): string => (id ? (PLAYER_COLORS[id] ?? UNKNOWN) : UNKNOWN);
export const labelOf = (id: PlayerId | null | undefined): string => (id ? (PLAYER_LABELS[id] ?? id) : "未知");

export function jobProgress(stages: { progress: number; state: string }[]): number {
  const counted = stages.filter((s) => s.state !== "skipped");
  if (!counted.length) return 0;
  return counted.reduce((a, s) => a + (s.state === "succeeded" ? 1 : s.progress), 0) / counted.length;
}
