"""Hit sound: onset strength of the video's audio around a hit candidate.

A racket hitting a shuttle makes a short broadband "pop". Onset strength is a spectral flux on a log-magnitude STFT with a
3-bin maximum filter on the previous frame (SuperFlux, Böck & Widmer, DAFx 2013), summed over bins above 2 kHz where the
pop dominates the hall noise. ``Onsets.features(t_ms)`` gives the scorer four numbers: the peak onset strength just after the
candidate relative to the local median (the sound arrives ~80 ms after the labelled contact frame on our footage: video
timing, not physics), its time offset, how many strong onsets surround it (busy halls, other courts), and whether audio exists.
Computed with numpy only; no audio (or no file) gives the neutral row ``NO_AUDIO``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

SR = 22050
N_FFT, HOP = 1024, 110  # 5 ms hop
HI_HZ = 2000.0
LAG_S = 0.08  # typical audio lag of the contact on our footage (median over 414 labelled hits)
NO_AUDIO = (0.0, 0.0, 0.0, 0.0)


def onset_strength(x: np.ndarray, sr: int = SR) -> tuple[np.ndarray, np.ndarray]:
    """``(t_s, odf)`` for mono float samples ``x``."""
    win = np.hanning(N_FFT).astype(np.float32)
    nf = max((len(x) - N_FFT) // HOP, 0)
    hi_bin = int(HI_HZ / sr * N_FFT)
    odf = np.zeros(nf, np.float32)
    prev = None
    for s in range(0, nf, 4096):
        e = min(nf, s + 4096)
        idx = np.arange(s, e)[:, None] * HOP + np.arange(N_FFT)[None, :]
        S = np.log1p(10 * np.abs(np.fft.rfft(x[idx] * win, axis=1)))[:, hi_bin:]
        if prev is None:
            prev = S[:1]
        P = np.vstack([prev, S[:-1]])
        P = np.maximum(np.maximum(P, np.roll(P, 1, 1)), np.roll(P, -1, 1))
        odf[s:e] = np.maximum(S - P, 0).sum(1)
        prev = S[-1:]
    t = (np.arange(nf) * HOP + N_FFT / 2) / sr
    return t.astype(np.float32), odf


def read_audio(video: Path, sr: int = SR) -> np.ndarray | None:
    """Mono float32 samples of ``video``'s first audio track via ffmpeg, or None without audio."""
    try:
        r = subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-vn", "-ac", "1", "-ar", str(sr), "-f", "s16le", "-"],
                           capture_output=True, check=False, timeout=3600)
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode != 0 or not r.stdout:
        return None
    return np.frombuffer(r.stdout, np.int16).astype(np.float32) / 32768.0


class Onsets:
    def __init__(self, t: np.ndarray, odf: np.ndarray):
        self.t = np.asarray(t, np.float32)
        self.odf = np.asarray(odf, np.float32)
        # local background: median over 2 s blocks (robust to a single loud event)
        blk = max(int(2.0 / (HOP / SR)), 1)
        nb = max(len(self.odf) // blk, 1)
        med = np.array([np.median(self.odf[i * blk:(i + 1) * blk]) for i in range(nb)], np.float32)
        self.bg = np.maximum(np.repeat(med, blk)[: len(self.odf)], 1e-4)
        if len(self.bg) < len(self.odf):
            self.bg = np.concatenate([self.bg, np.full(len(self.odf) - len(self.bg), self.bg[-1] if len(self.bg) else 1e-4)])
        self.ratio = self.odf / self.bg

    @classmethod
    def from_npz(cls, path: Path) -> "Onsets":
        d = np.load(path)
        return cls(d["t"], d["hi"] if "hi" in d else d["odf"])

    @classmethod
    def from_video(cls, video: Path) -> "Onsets | None":
        x = read_audio(video)
        if x is None or len(x) < N_FFT * 2:
            return None
        return cls(*onset_strength(x))

    def features(self, t_ms: float) -> tuple[float, float, float, float]:
        """``(log1p peak ratio in [t, t+2*LAG], (argmax - t - LAG)/LAG, strong onsets in +-0.5 s / 10, 1)``."""
        t = t_ms / 1000.0
        i0, i1 = np.searchsorted(self.t, [t - 0.02, t + 2 * LAG_S])
        if i1 <= i0:
            return NO_AUDIO
        k = i0 + int(np.argmax(self.ratio[i0:i1]))
        peak = float(self.ratio[k])
        j0, j1 = np.searchsorted(self.t, [t - 0.5, t + 0.5])
        r = self.ratio[j0:j1]
        n_strong = int(np.sum((r[1:-1] > 4.0) & (r[1:-1] >= r[:-2]) & (r[1:-1] >= r[2:]))) if len(r) > 2 else 0
        return (float(np.log1p(peak)), float(np.clip((self.t[k] - t - LAG_S) / LAG_S, -2, 2)), min(n_strong, 10) / 10.0, 1.0)
