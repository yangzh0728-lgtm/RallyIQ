import { useSyncExternalStore } from "react";
import type { PlaybackClock } from "./clock";

/** Re-render on every clock tick and return the current time in ms. */
export function useClockMs(clock: PlaybackClock): number {
  return useSyncExternalStore(
    (cb) => clock.subscribe(cb),
    () => clock.ms,
  );
}

export function useClockPlaying(clock: PlaybackClock): boolean {
  return useSyncExternalStore(
    (cb) => clock.subscribe(cb),
    () => clock.playing,
  );
}

/** The playback rate; the clock emits on setRate, so this also follows changes made while paused. */
export function useClockRate(clock: PlaybackClock): number {
  return useSyncExternalStore(
    (cb) => clock.subscribe(cb),
    () => clock.rate,
  );
}
