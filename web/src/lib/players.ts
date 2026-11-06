import { PLAYER_COLORS, PLAYER_LABELS, TEAM_COLORS, TEAM_LABELS, type PlayerId, type TeamId } from "./types";

const UNKNOWN = "#64748b";
const isTeam = (id: string): id is TeamId => id === "A" || id === "B";
export const colorOf = (id: PlayerId | TeamId | string | null | undefined): string =>
  id ? (isTeam(id) ? TEAM_COLORS[id] : (PLAYER_COLORS[id as PlayerId] ?? UNKNOWN)) : UNKNOWN;
export const labelOf = (id: PlayerId | TeamId | string | null | undefined): string =>
  id ? (isTeam(id) ? TEAM_LABELS[id] : (PLAYER_LABELS[id as PlayerId] ?? id)) : "未知";

export function jobProgress(stages: { progress: number; state: string }[]): number {
  const counted = stages.filter((s) => s.state !== "skipped");
  if (!counted.length) return 0;
  return counted.reduce((a, s) => a + (s.state === "succeeded" ? 1 : s.progress), 0) / counted.length;
}
