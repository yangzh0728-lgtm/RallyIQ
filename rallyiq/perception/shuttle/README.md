# 羽毛球检测与轨迹追踪（stage `shuttle`）

输入一段视频，输出 `ShuttleTrack`（见 `contracts/`）：每帧最多一个球点，带置信度；
`source` 区分模型检测（`detected`）和补点（`interpolated`）；看不到球的帧记为 `gaps`，
并标注 `out_of_frame`（球出画）或 `not_detected`。

## 安装与权重

```bash
pip install -e '.[shuttle]'          # torch + OpenCV
pip install gdown && python scripts/fetch_shuttle_weights.py   # 下载到 weights/shuttle/heatmap.pt
```

权重查找顺序：`ctx.config["shuttle"]["weights"]` → 环境变量 `RALLYIQ_SHUTTLE_WEIGHTS`
→ 仓库 `weights/shuttle/` → `~/.cache/rallyiq/weights/shuttle/`。
模型结构和预训练权重为 MIT 许可（`LICENSE-THIRD-PARTY`）。

没有权重时，stage 返回 `detector=null`、没有点的轨迹（下游应当理解为"未分析"，而不是"球一直不可见"）；
设置 `require_weights: true` 则直接报错。

设备自动选择 CUDA → Apple MPS → CPU，可用 `device` 覆盖。

## 单独运行

```bash
python -m rallyiq.perception.shuttle clip.mp4 --out shuttle.json --candidates cands.json
# 输出评估基准的预测文件（只含真实检测，补点不计入）
python -m rallyiq.perception.shuttle clip.mp4 --benchmark-clip-id <id> \
    --benchmark-start-ms <片段起点毫秒> --benchmark-out predictions/<id>.json
```

结束时打印处理速度（帧/秒）和检测率。

## 做法

1. **检测（`detector.py`）**：多帧热力图网络，一次看 8 帧，加上每个机位段的背景中值图（抑制静止亮点）。
   窗口逐帧滑动，每帧的热力图是所有包含它的窗口的加权平均（时间集成）。
   视频逐帧流式读取，不会整段载入内存。
2. **候选（`heatmap.py`）**：每帧保留最多 3 个热力图峰值，阈值放低到 0.2（常规做法是 0.5 且只取最大块），
   用亚像素质心。远端小球的峰值往往只有 0.2–0.4，交给追踪器靠时间连续性来判断。
3. **追踪（`tracker.py`，只依赖 numpy）**：
   - 按匀速预测和随速度变化的门限逐帧关联成小段；
   - 按总分从高到低挑选小段，每帧只保留一个点；弱候选只有连成 ≥3 帧的一致小段才保留；
   - 去掉孤立跳点，但保留击球时的方向突变；
   - 短缺口（≤0.3 s）用二次曲线补点，拟合不好时只对 ≤0.1 s 的缺口线性补点；补点置信度更低，且不补到画面外；
   - 长缺口保持为缺口，不伪造轨迹。
4. **原始候选不丢失**：所有候选（包括被拒绝的）写入 `<work_dir>/shuttle_candidates.json`，
   标明哪个被采用，便于复核和积累难例。

### 远端小球与低帧率

- 低阈值 + 时间一致性（上面第 2、3 步）。
- `roi: auto`：如果有球场标定，把画面裁到球场及其上方空域（16:9），只在能放大 ≥1.25 倍时启用，
  让远端小球在网络输入里更大。**实验性，默认关闭**，需要在评估集上对比后再决定是否默认打开。
- 所有阈值按秒和画面对角线比例设定，25/30/60/120 fps、720p–4K 用同一套参数。
- 高帧率视频（如 120 fps）拆成约 30 fps 的交错子序列分别推理（0,4,8… / 1,5,9…），
  让网络看到的帧间运动接近训练条件；每帧仍只处理一次。

## 配置（`ctx.config["shuttle"]`）

| 键 | 默认 | 说明 |
| --- | --- | --- |
| `weights` | 自动查找 | 权重文件或目录 |
| `require_weights` | `false` | 没有权重时报错 |
| `device` | 自动 | `cuda` / `mps` / `cpu` |
| `batch_size` | 8 | |
| `peak_threshold` | 0.2 | 候选峰值阈值 |
| `top_k` | 3 | 每帧候选数 |
| `target_fps` | 30 | 高帧率拆分目标 |
| `ensemble` | `weight` | `weight` / `average` / `none` |
| `roi` | `off` | `auto` 启用球场区域裁剪 |
| `tracker` | `{}` | 覆盖 `TrackerConfig` 字段 |
