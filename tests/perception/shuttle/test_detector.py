"""Detector plumbing tests. Need torch + OpenCV (``pip install -e '.[shuttle]'``); skipped otherwise.

There are no real weights in CI, so the network is replaced by a stand-in that
paints a Gaussian wherever the frame is brightest. That checks everything around
the network: frame streaming, windowing, temporal ensemble, high-fps stream
splitting, ROI cropping and coordinate mapping back to original pixels.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
cv2 = pytest.importorskip("cv2")

from rallyiq.perception.shuttle import detector as D  # noqa: E402
from rallyiq.perception.shuttle.model import INPUT_HEIGHT, INPUT_WIDTH, build_heatmap_net  # noqa: E402


def ball_position(f, w, h):
    return (0.1 * w + f * 0.8 * w / 60.0, 0.3 * h + 0.2 * h * np.sin(f / 10.0))


def write_video(path, n=60, w=640, h=360, fps=30.0):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in range(n):
        img = np.full((h, w, 3), 40, np.uint8)
        x, y = ball_position(f, w, h)
        cv2.circle(img, (int(round(x)), int(round(y))), 3, (255, 255, 255), -1)
        vw.write(img)
    vw.release()
    return path


class BrightestSpotNet(torch.nn.Module):
    """Stand-in network: Gaussian at the brightest pixel of every input frame."""

    def __init__(self, seq_len, bg):
        super().__init__()
        self.seq_len, self.bg = seq_len, bg
        self.calls = []

    def forward(self, x):
        n = x.shape[0]
        frames = x[:, 3:] if self.bg else x
        frames = frames.reshape(n, self.seq_len, 3, INPUT_HEIGHT, INPUT_WIDTH).mean(2)
        out = torch.zeros(n, self.seq_len, INPUT_HEIGHT, INPUT_WIDTH)
        yy, xx = torch.meshgrid(torch.arange(INPUT_HEIGHT), torch.arange(INPUT_WIDTH), indexing="ij")
        for i in range(n):
            for k in range(self.seq_len):
                idx = int(frames[i, k].argmax())
                cy, cx = divmod(idx, INPUT_WIDTH)
                out[i, k] = torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 2.5 ** 2))
        self.calls.append(n)
        return out


def make_detector(tmp_path, **cfg):
    net = build_heatmap_net(8, "concat")
    ckpt = tmp_path / "heatmap.pt"
    torch.save({"model": net.state_dict(), "param_dict": {"seq_len": 8, "bg_mode": "concat"}}, ckpt)
    det = D.HeatmapShuttleDetector(ckpt, D.DetectorConfig(device="cpu", **cfg))
    det.net = BrightestSpotNet(8, True)
    return det


def test_real_checkpoint_layout_loads(tmp_path):
    net = build_heatmap_net(8, "concat")
    assert net(torch.zeros(1, 27, INPUT_HEIGHT, INPUT_WIDTH)).shape == (1, 8, INPUT_HEIGHT, INPUT_WIDTH)
    det = make_detector(tmp_path)
    assert det.seq_len == 8 and det.bg_mode == "concat" and det.name.startswith("shuttle-heatmap-seq8@")


@pytest.mark.parametrize("fps,n,ensemble", [
    (30.0, 60, "weight"), (120.0, 61, "weight"), (30.0, 5, "weight"), (30.0, 61, "none"), (60.0, 45, "average"),
])
def test_every_frame_detected_and_mapped_back(tmp_path, fps, n, ensemble):
    video = write_video(tmp_path / "v.mp4", n=n, fps=fps)
    det = make_detector(tmp_path, batch_size=4, ensemble=ensemble)
    cands, size = det.detect(video, fps, n)
    assert size == (640, 360)
    assert sorted(cands) == list(range(n))
    for f in range(n):
        x, y = ball_position(f, 640, 360)
        cx, cy, s = cands[f][0]
        assert abs(cx - x) < 2.5 and abs(cy - y) < 2.5, (f, cx, cy, x, y)
        assert s > 0.9


def test_roi_crop_maps_back_to_original_pixels(tmp_path):
    video = write_video(tmp_path / "v.mp4", n=20)
    det = make_detector(tmp_path)
    roi = (32, 40, 32 + 576, 40 + 324)
    cands, _ = det.detect(video, 30.0, 20, [D.Segment(0, 19, roi)])
    for f in range(20):
        x, y = ball_position(f, 640, 360)
        cx, cy, _ = cands[f][0]
        assert abs(cx - x) < 2.5 and abs(cy - y) < 2.5


def test_find_weights(tmp_path, monkeypatch):
    monkeypatch.delenv("RALLYIQ_SHUTTLE_WEIGHTS", raising=False)
    (tmp_path / "heatmap.pt").write_bytes(b"x")
    assert D.find_weights(tmp_path) == tmp_path / "heatmap.pt"
    monkeypatch.setenv("RALLYIQ_SHUTTLE_WEIGHTS", str(tmp_path / "heatmap.pt"))
    assert D.find_weights() == tmp_path / "heatmap.pt"


def test_ensemble_weights_centre_heavy():
    w = D.ensemble_weights(8, "weight")
    assert abs(w.sum() - 1) < 1e-6 and w[3] == w[4] == w.max() and w[0] == w[7] == w.min()
