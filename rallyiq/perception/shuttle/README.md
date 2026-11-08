# 羽毛球检测与轨迹追踪（stage `shuttle`）

输入一段视频，输出 `ShuttleTrack`（见 `contracts/`）：每帧最多一个球点，带置信度；
`source` 区分模型检测（`detected`）和补点（`interpolated`）；看不到球的帧记为 `gaps`，
并标注 `out_of_frame`（球出画）或 `not_detected`。

## 安装与权重

```bash
pip install -e '.[shuttle]'          # torch + OpenCV
bash scripts/fetch_shuttle_weights.sh   # 下载到 weights/shuttle/heatmap.pt（mac_e2e.sh 会自动调用）
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
   追踪器的弱候选门限 `weak_score` 跟随这个阈值（除非在 `tracker` 里单独指定），
   否则调低检测阈值只会多出被追踪器丢掉的候选。
   2026-11-06 在 4 段真实片段上试过 0.1（`scripts/shuttle_sweep.py`、`scripts/hits_ablation.py`，无标注统计）：
   跟踪覆盖提高（副场 ko26_court3 50% → 66%，主转播 ae26_ms_f 69% → 76%），但新增的低分点里有不少噪声，
   制造假的方向变化：击球候选变多（court3 47 → 78），高置信击球（ok）在每段都减少（ae26 27 → 21，court3 12 → 5–7，
   wc25 10 → 5），调低事件阶段的 `min_shuttle_conf` 也补不回来。所以默认保持 0.2；
   要提高覆盖，需要让低分点不进入击球判断，或者在标注数据上微调模型，不能只调低阈值。
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
  让远端小球在网络输入里更大。**默认关闭**：真实片段上，副场固定机位的球场占画面宽度 61%，不触发裁剪；
  主转播只多 2 个百分点。
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
| `peak_threshold` | 0.2 | 候选峰值阈值，追踪器的弱候选门限随之变化 |
| `top_k` | 3 | 每帧候选数 |
| `target_fps` | 30 | 高帧率拆分目标 |
| `ensemble` | `weight` | `weight` / `average` / `none` |
| `roi` | `off` | `auto` 启用球场区域裁剪 |
| `tracker` | `{}` | 覆盖 `TrackerConfig` 字段 |
| `court_lock` | `auto` | 多块场地同框时只保留目标场地上空的轨迹；`off` 关闭 |

## 多场地：锁定目标场地

一个画面里有几场比赛时，热力图模型会把别的场地的球也检出来，而且常常更清楚。
`court_lock`（默认开）按**整条轨迹**判断归属，见 `rallyiq/perception/target_court.py`：

- 有标定时：目标场地（含界外缓冲）上方的"空中区域"在画面里的投影，高度上限随位置变化
  （场地中部最高 10 m，底线和边线外更低）。标定列出的其他场地上方的点权重为 0；
  和其他场地在画面上重叠的部分算"有歧义"。
- 没有标定时：用两名球员覆盖的横向范围代替。
- 轨迹的平均权重低于 0.5 整条丢掉；主要处在歧义区的轨迹，必须经过本场球员的击球范围才保留；
  经过本场球员的轨迹在选择时加分。

合成评测（`python -m evaluation.multicourt`）：左右相邻场地串场率从 57–73% 降到 0%，召回不降；
前后相邻（邻场在远端底线后面）是单目视角的难点，串场率从 95% 降到约 23%。

## 微调（`training.py`、`scripts/finetune_shuttle.py`）

预训练模型在副场固定机位（高位俯拍、球在 512×288 输入里约 3 px）上只有约 50% 的覆盖，
光调阈值会引入噪声（见上文）。所以要在自己标注的片段上微调。流程：

1. **预标注**：`scripts/real_run.py` 会写出 `preds/<clip>.json`。
2. **只修可疑帧**：
   `python -m evaluation.tools.label_tool ko26_court3 --prefill ~/rallyiq-data/runs/latest/preds/ko26_court3.json`，
   按 `2` 进入羽毛球模式，再用 `g` / `G` 在可疑帧之间跳：低分检测、≤15 帧的短缺口、长缺口两端各 3 帧。
   其余帧的预标注一般是对的，抽查几段即可。修完按 `r` 标成已复核（reviewed），按 `q` 保存。
3. **训练**（Mac 上用 MPS）：
   ```bash
   # 只有一段副场标注时：前 70% 训练，后 30% 验证
   python scripts/finetune_shuttle.py --split-clip ko26_court3:0.7 --out weights/shuttle/side.pt
   # 有 split=train 的标注片段时：用它们训练，用已标注的 eval 片段验证
   python scripts/finetune_shuttle.py --out weights/shuttle/finetuned.pt
   ```
   每轮会打印验证集在阈值 0.5 / 0.3 / 0.2 下的精确率、召回和 F1，第 0 轮是预训练模型本身，
   最后按 0.2（检测器实际用的阈值）下的 F1 保留最好的一轮。按 0.5 选曾经选出一个在 0.2 下更差的轮次。checkpoint 和检测器的格式一样，旁边会写一个同名的 `.json` 报告。
4. **对比**：`RALLYIQ_SHUTTLE_WEIGHTS=weights/shuttle/side.pt python scripts/real_run.py --only ko26_court3`，
   再用 `scripts/hits_ablation.py` 看 ok 击球有没有一起变好，不要只看覆盖率。

默认只用 `status: reviewed` 的标注：未经复核的预标注拿来训练，等于让模型学自己的错误。
默认设置是 BatchNorm 冻结、学习率 1e-4、10 轮、每轮 300 步、batch 4，数据增强只有水平翻转和亮度/对比度抖动。

### 持续训练（`scripts/shuttle_loop.py`）

日常用这一个命令就够了，每标完一段就跑一次：

```bash
python scripts/shuttle_loop.py            # 训练一轮，和现役模型比，变好才替换
python scripts/shuttle_loop.py --status   # 现役模型、历次结果、下一段该标什么
python scripts/shuttle_loop.py --rollback # 回到上一个模型
```

每一轮：所有已复核片段按时间切开（前 70% 训练、后 30% 验证；`--holdout ID` 让整段只做验证，
会被记住），从原始预训练权重开始微调（`--from champion` 改为接着现役模型练），然后把现役模型
放在**同一批**验证帧上评一次。阈值 0.2 下 F1 至少高 0.01、精确率最多降 0.03，才采用新模型，
并复制为 `weights/shuttle/heatmap.pt`（流水线默认加载它，除非设了 `RALLYIQ_SHUTTLE_WEIGHTS`）。
所有模型、`state.json` 和 `history.md` 都在 `weights/shuttle/loop/`。标注没变化时不会重复训练
（`--force` 强制）。采用后脚本会打印在没参与训练的片段上跑 `real_run.py` 和 `hits_ablation.py`
的命令：验证分数是逐帧检测，击球结果还要单独核对。

**标注检查。** 用预标注打底时，模型没检测到的每一帧都会被标成"看不见"，其中有些球其实看得见
（比如发球前在手里）。这些帧会让训练学错、让验证把对的检测算成误报。所以每轮结束后，脚本用
最后一轮（学得最多的）模型过一遍所有已标注帧，把和标注对不上的帧写到
`weights/shuttle/loop/round-NNN-check.json`，并拼一张最有把握的 48 帧的图（`.jpg`，绿圈=标注，
紫圈=模型）。然后用
`python -m evaluation.tools.label_tool <clip> --check weights/shuttle/loop/round-NNN-check.json`
只看这些帧：`g` 跳过（标注是对的），`y` 采用模型的点，左键点击或 `n` 改标注，`x` 删掉拿不准的帧
（训练时忽略）。`--check WEIGHTS` 不训练，只对任意 checkpoint 做这项检查。

**自动标注（`scripts/auto_label.py`）。** 不用人从头标：以流水线的预测（或已有标注）为起点，
用最近一轮的末轮模型跑遍整段视频，逐帧决定：手工改过的帧不动；模型和标注一致的帧不动；标注说看不见、
模型以 ≥0.4 的分数看到并且前后 ±2 帧有呼应的点，**提议**为可见；其余矛盾的帧去掉标注（训练时忽略，
不让模型学一个猜测）。每个提议在拼图里有编号（2 倍放大，紫圈=提议的点），看完后用
`--apply --not-shuttle 编号 --unsure 编号` 写入（状态 reviewed，`auto_label` 字段记录模型、计数和采用的帧），
`--preview` 生成带标注圈和帧号的预览视频，方便截图指出问题。输出在 `~/rallyiq-data/autolabel/<clip>/`。
