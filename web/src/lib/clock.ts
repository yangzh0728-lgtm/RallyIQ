/**
 * Single playback clock for the workbench. Backed by the <video> element when there is one,
 * otherwise by a virtual timer (mock demo with no file), so overlays/timeline work either way.
 */
export class PlaybackClock {
  private video: HTMLVideoElement | null = null;
  private virtualMs = 0;
  private virtualPlaying = false;
  private lastTick = 0;
  private raf = 0;
  private listeners = new Set<() => void>();
  rate = 1;

  constructor(public durationMs: number) {}

  private readonly onMedia = () => this.emit();
  private static readonly MEDIA_EVENTS = ["play", "pause", "seeked", "ended", "ratechange"];

  attach(video: HTMLVideoElement | null) {
    this.detach();
    this.video = video;
    if (video) {
      video.playbackRate = this.rate;
      video.currentTime = this.virtualMs / 1000;
      PlaybackClock.MEDIA_EVENTS.forEach((e) => video.addEventListener(e, this.onMedia));
    }
    this.raf = requestAnimationFrame(this.loop);
  }

  detach() {
    cancelAnimationFrame(this.raf);
    if (this.video) PlaybackClock.MEDIA_EVENTS.forEach((e) => this.video!.removeEventListener(e, this.onMedia));
    if (this.video) this.virtualMs = this.video.currentTime * 1000;
    this.video = null;
  }

  get ms(): number {
    return this.video ? this.video.currentTime * 1000 : this.virtualMs;
  }

  get playing(): boolean {
    return this.video ? !this.video.paused && !this.video.ended : this.virtualPlaying;
  }

  play() {
    if (this.video) void this.video.play();
    else {
      if (this.virtualMs >= this.durationMs) this.virtualMs = 0;
      this.virtualPlaying = true;
      this.lastTick = performance.now();
    }
    this.emit();
  }

  pause() {
    if (this.video) this.video.pause();
    else this.virtualPlaying = false;
    this.emit();
  }

  toggle() {
    if (this.playing) this.pause();
    else this.play();
  }

  seek(ms: number) {
    const t = Math.min(Math.max(0, ms), this.durationMs);
    if (this.video) this.video.currentTime = t / 1000;
    this.virtualMs = t;
    this.emit();
  }

  setRate(rate: number) {
    this.rate = rate;
    if (this.video) this.video.playbackRate = rate;
    this.emit();
  }

  subscribe(fn: () => void): () => void {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }

  private emit() {
    this.listeners.forEach((l) => l());
  }

  private loop = () => {
    if (!this.video && this.virtualPlaying) {
      const now = performance.now();
      this.virtualMs += (now - this.lastTick) * this.rate;
      this.lastTick = now;
      if (this.virtualMs >= this.durationMs) {
        this.virtualMs = this.durationMs;
        this.virtualPlaying = false;
        this.emit();
      }
    }
    if (this.playing) this.emit();
    this.raf = requestAnimationFrame(this.loop);
  };
}
