# 音频 / 音视频击球检测调研（RallyIQ 第 2 轮）

> 注：原始调研稿。许可证和数字已另行逐条核查，更正见 `../2026-11-28-web-research-round2.md` 第三节和 `verification.json`；不一致时以核查为准。


> 日期：2026-11-28 · 范围：拍球接触（hit）的音频检测、音视频融合、多场地串音、手机/运动相机音画偏移、可商用库，以及给 RallyIQ 击球检测器加一路"音频候选流"的落地配方。
> 已调研过的条目（TrackNet 系列、WASB、AdaSpot、E2E-Spot、Padel CVPRW 2024 多模态、MatchClip-AI 音频思路等）不再重复。
> 许可证均读自 LICENSE 文件、PyPI 元数据或数据集/论文页面；"无 LICENSE 文件"按保留所有权利处理。读不到的地方都明确标为"未核实"。

---

## 0. 结论速览

1. **羽毛球没有公开、可商用的"击球声"数据集或论文代码。** AudioSet 本体里没有 badminton/racket 类（最接近的是 `Whack, thwack`、`Slap, smack`、`Tap`、`Clicking`、`Squeak`，已在 ontology.json 中核对）。网上能找到的羽毛球音频仓库都是业余项目（无 LICENSE，没报精度）。所以 RallyIQ 只能**自建**：先用传统 DSP 起音检测（无需训练），再用视觉高置信击球做自监督标签，训练一个小的"本场击球 / 非本场"判别器。
2. **最有价值的用法不是"音频单独找击球"，而是三件事：**
   (a) **时间精修**：击球声在接触瞬间发出，起音可以定位到 1–5 ms。视觉拐点在 30 fps 下有 ±1 帧误差，还会被平滑带偏；
   (b) **确认 / 打分特征**：给 `learned.py` 的打分器加"附近有没有匹配的起音"等特征；
   (c) **找回漏检**：视觉在接触帧丢了球（低机位下，近端球员身体常挡住球）时，用音频起音提出新候选，再用姿态和轨迹交叉核对。
3. **多场地串音是主要风险。** 估算：邻场合计每秒约 2 个外来起音时，任意 ±25 ms 窗口里碰巧落进一个外来起音的概率约 10%。所以"音频确认视觉候选"是安全的，但"**纯音频提候选**"在多场馆里必须有姿态和轨迹佐证。推荐每段视频做一次**自校准**：正样本是与视觉高置信击球对上的起音，负样本是本场回合间歇里的起音（这时邻场还在打），用它们训练一个"本场声学模型"。
4. **音画偏移必须逐视频估计，不能假设为 0。** 偏移有三个来源：容器层的 start_time / edit list 偏移；手机可变帧率（RallyIQ 目前按平均帧率换算时间戳，`ingest/quality.py` 已经标出这个风险）；声速传播延迟。声速延迟在底线后机位下，远端击球比近端晚约 30 ms，相当于 30 fps 下 1 帧、60 fps 下 2 帧，可以用标定得到的相机位置和击球点距离补偿。最稳的办法是 ICASSP 2021 Ebenezer 等人的思路：两路各自检测击球事件，再做互相关，求出偏移。
5. **许可证**：librosa（ISC）、scipy（BSD）、soundfile（BSD-3）、PANNs 代码（MIT）+ 权重（Zenodo，CC BY 4.0）、PretrainedSED 代码 + 权重（MIT）、EfficientAT（MIT）、Synchformer（MIT）、pyroomacoustics（MIT）都可以商用。要避开的有：**madmom 的模型文件（CC BY-NC-SA 4.0，只有代码是 BSD）**、**aubio（GPL-3）**、**essentia（AGPL-3）**、**TT Sounds 数据（CC BY-NC 4.0）**。
6. **预期收益（估算，未实测）**：最接近的公开证据是网球转播上的音频加球轨迹融合（Huang & Cox 2011）：F 值从仅视频的 67.6–70.6 提到 77.0–84.2。在单场地业余视频上，召回率可能从学习打分器的 34/58（规则 47/58）提到约 48–53/58，同时精度基本不变；命中击球的时间误差中位数从约 1 帧降到半帧以内。多场馆里的增益要打折扣。第 7 节给出 1 天就能做完的 go/no-go 实验。

---

## 1. 物理与信号约束（直接决定参数）

| 量 | 数值 | 对 RallyIQ 的含义 |
|---|---|---|
| 声速 | 约 343 m/s（20 °C；温度每升 1 °C 约 +0.6 m/s） | 每米约 2.9 ms |
| 底线后机位：近端击球距离 | 约 3–6 m → **9–18 ms** | 30 fps 下不到 1 帧 |
| 底线后机位：远端击球距离 | 约 14–17 m（场长 13.4 m + 机位后撤） → **41–50 ms** | 比近端晚约 30 ms：30 fps 下 1 帧，**60 fps 下 2 帧** |
| 侧面机位 | 约 3–9 m → 9–26 ms | 差异小，但仍需补偿 |
| 帧间隔 | 30 fps = 33.3 ms；60 fps = 16.7 ms | 匹配窗口取 ±max(25 ms, 1 帧) |
| 手机音频 | 一般是 AAC 48 kHz（Hear the Sweet Spot 一文用 iPhone 15 Plus，44.1/48 kHz 即可用） | 起音定位可到 1 ms 以内；AAC 预回声会把起音抹开约几 ms，可以忽略 |
| 羽毛球最短击球间隔 | 双打平抽挡对攻约 0.25–0.35 s（经验值） | 不应期取 120 ms，不能照搬网球的 250 ms |
| 主要干扰声 | 鞋底摩擦声（AudioSet `Squeak`）、球落地、喊叫、拍框相碰、邻场击球、球馆混响（大馆 RT60 常有 1.5–3 s，经验值） | 带通加自适应阈值，再加一个第二级分类器 |

**RallyIQ 现状（读代码得到）：** `rallyiq/pipeline/probe.py:19` 只探测 `-select_streams v:0`，整个代码库没有任何音频代码。帧时间戳用 `meta.t_ms(frame)` 按平均帧率换算；`rallyiq/ingest/quality.py:207` 已经对 VFR 给出"个别时刻可能偏差几帧"的提示。`rallyiq/perception/calibration/homography.py:262` 已经在搜索焦距，所以从 H 和 K 分解出相机地面位置、进而算出声源距离是可行的。

---

## 2. Top 5（按对 RallyIQ 的落地价值排序）

| # | 条目 | 许可证（代码 / 权重 / 数据） | 关键数字 | 怎么接进 RallyIQ |
|---|---|---|---|---|
| 1 | **DSP 起音栈：librosa SuperFlux 起音 + scipy 零相位带通 / 自适应 MAD 阈值**（参考实现：tennis-hit-clipper） | librosa ISC；scipy BSD；tennis-hit-clipper MIT | 无公开精度；tennis-hit-clipper 的参数：1.5–8 kHz 带通、10 ms RMS、8 s 滑窗"中位数 + k·MAD"、250 ms 不应期。同类的能量峰法在 TT Sounds 上 P/R 为 0.98/0.95–1.00，起音误差 ≤ 0.2 ms（近距离麦克风） | 新建 `rallyiq/audio/onsets.py`，输出每个起音的时间、SNR、频带能量比和立体声 ITD；是第 7 节配方的第 1 步，不需要训练 |
| 2 | **Huang & Cox 等，AVSP 2011：用多模态信息改进网球击球检测**（音频击球声加视觉球轨迹融合） | 只有论文；**无代码、无数据** | 澳网 2008 女双 4 段共 1086 次击球：仅音频 F 73.5–77.3，仅视频 F 67.6–70.6，**融合 F 77.0–84.2**；音频窗口为视觉击球 ±8 帧；文中提到拍到麦克风约 30 ms 延迟 | 融合规则几乎可以原样搬过来：以视觉击球时刻为中心的高斯窗约束音频击球，再加击球间隔先验。对应第 7 节第 4 步；它也是第 8 节收益估算的主要外部依据 |
| 3 | **Ebenezer 等，ICASSP 2021：靠事件检测发现音画不同步**（网球击球） | 论文 arXiv 页标 CC BY-NC-SA 4.0；**无代码、无数据** | 6 h 网球视频，约 50.4 万帧，模拟偏移；摘要说"high accuracy"，没给数字 | 方法直接搬用：两路各自检测击球，用互相关估出每段视频的 Δ（第 7 节第 2 步） |
| 4 | **PretrainedSED**（Schmid 等，ICASSP 2025）帧级 SED 预训练模型，含 1.6M 参数的 `frame_mn06` | 代码 MIT；checkpoint 在同仓库 GitHub Release（MIT 仓库）；预训练数据 AudioSet（见第 6 节注意事项） | AudioSet Strong 帧级预训练；README 说 DESED 设置下 PSDS1 约 46 | 第 2 阶段：用视觉自监督标签微调成"本场击球"帧级判别器，或者在起音处取嵌入向量作为打分特征 |
| 5 | **PANNs**（Kong 等，TASLP 2020）+ `panns_inference` | 代码 MIT；权重 Zenodo 3987831 为 **CC BY 4.0** | Cnn14 在 AudioSet 上 mAP 0.431；SED 版 Cnn14_DecisionLevelMax mAP 0.385，可输出帧级结果 | 在起音附近 0.5 s 窗口取 2048 维嵌入，以及 `Whack/Slap/Tap/Squeak/Speech` 的帧级概率，作为第二级分类器的特征 |

次选：Baughman 等 MMSports 2019（网球声学事件，F1 92.39%，提供第二级分类器的结构和负类清单）、TT Sounds（CC BY-NC，只能作研究参考，但给出了毫秒级能量峰检测的具体参数）、Synchformer（MIT，用于粗偏移体检）。

---

## 3. 条目详单

### 3.1 librosa 起音检测（SuperFlux / 谱通量）+ scipy
- **链接**：https://github.com/librosa/librosa ；scipy：https://github.com/scipy/scipy
- **许可证**：librosa **ISC**（GitHub 与 PyPI 均为 ISC）；scipy **BSD**；soundfile **BSD-3**（依赖 libsndfile，LGPL，动态链接即可）。
- **精度**：通用库，没有羽毛球数字。`librosa.onset.onset_strength(lag, max_size)` 能实现 Böck & Widmer 的 SuperFlux（用最大值滤波压制颤音和混响）；`onset_detect(backtrack=True)` 可以回溯到能量起点。
- **接入**：RallyIQ 已经依赖 ffmpeg；用 `ffmpeg → 48 kHz PCM → scipy.signal.sosfiltfilt`（零相位，不引入群延迟）做带通，再做 RMS 包络和 SuperFlux，最后 `find_peaks`。这是不需要训练的第一路候选流，跑一段视频只要几秒 CPU。

### 3.2 tennis-hit-clipper（业余网球音频击球切片器）
- **链接**：https://github.com/jhsgumban/tennis-hit-clipper
- **许可证**：**MIT**（GitHub 元数据）。
- **精度**：README 没报。方法：ffmpeg 转 16 kHz 单声道 → scipy 带通 1.5–8 kHz → 10 ms RMS 包络 → 8 s 滑窗"中位数 + k·MAD"自适应阈值（默认 k=6，远端轻击可降到 4）→ `find_peaks`，不应期 250 ms。README 明确说它**分不清**自己、对手和球落地的声音。
- **接入**：可以直接借用参数结构。羽毛球要改三处：采样率保持 48 kHz（16 kHz 会丢掉 8 kHz 以上的拍弦"脆音"）；不应期降到约 120 ms；带通上限放宽到 10–12 kHz，再用已确认击球做逐频带 SNR 统计来调。"分不清谁打的"由视觉和姿态补上。

### 3.3 羽毛球业余音频仓库（证明可行，但不能拿来用）
- **duzitong/badminton-video-editor**：https://github.com/duzitong/badminton-video-editor ，**无 LICENSE 文件，保留所有权利**。读了 `src/badminton_editor/tools.py`：在业余羽毛球整场视频上用 `librosa.onset.onset_detect(delta=0.3, wait=10)` 加 0.3 s 去抖找"击球"，再按超过 3 s 的间隔分成回合（示例输出是 966 s 视频里 351 次击球、67 个簇）。**没报精度**。
- **riven314/BadmintonAudio-FastAI**：https://github.com/riven314/BadmintonAudio-FastAI ，**无 LICENSE 文件**，2020 年用 fastai 在转播羽毛球上训练击球声分类器，没给指标。
- **接入**：只能说明"业余羽毛球击球声用普通起音就能抓到大部分"，代码不能复用。另外，0.3 s 的去抖对双打平抽太长。

### 3.4 PretrainedSED（ICASSP 2025，帧级 SED 预训练）
- **链接**：代码 https://github.com/fschmid56/PretrainedSED ；论文 https://arxiv.org/abs/2409.09546
- **许可证**：仓库 **MIT**（LICENSE 文件已读）。checkpoint 由同仓库 GitHub Release v0.0.1 发布，没有另写许可证，视为 MIT。部分外部初始化权重另有来源（BEATs 在 OneDrive，属 unilm 的 MIT 仓库；ATST、M2D 链接到各自仓库）。
- **精度**：在 AudioSet Strong 帧级标注上做了额外预训练；README 写明 DESED 设置下 `val/psds1_macro_averaged` 约 46，不接序列模型时 41.8 PSDS1。新增的低复杂度模型 `frame_mn06`（1.62M 参数）和 `frame_mn10`（3.83M 参数）适合 CPU。README 也提醒：目标音频离 AudioSet 本体越远，增益越不确定。
- **接入**：第 2 阶段模型。以起音为中心取 0.5 s 片段，或者整段帧级推理，微调出两类头（本场击球 / 其他），训练标签来自第 7 节的自监督。也可以只冻结骨干取嵌入向量，喂给现有打分器。

### 3.5 PANNs + panns_inference
- **链接**：https://github.com/qiuqiangkong/audioset_tagging_cnn ；https://github.com/qiuqiangkong/panns_inference ；权重 https://zenodo.org/records/3987831 ；论文 https://arxiv.org/abs/1912.10211
- **许可证**：两个代码仓库都是 **MIT**；Zenodo 权重记录是 **CC BY 4.0**（页面已核对），需要署名。
- **精度**：Cnn14 AudioSet mAP 0.431（16 kHz 版 0.438），Wavegram-Logmel-Cnn14 0.439，SED 版 Cnn14_DecisionLevelMax 0.385；还有 MobileNetV1/V2（0.389/0.383）这类轻量版。
- **接入**：`panns_inference.SoundEventDetection` 直接给出 `framewise_output`（帧数 × 527 类）。在每个起音的 ±250 ms 内取 `Whack, thwack`、`Slap, smack`、`Tap`、`Squeak`、`Speech`、`Cheering` 的最大概率，再加上 Cnn14 的嵌入（可用 PCA 降到 16 维），作为"是不是击球声"的判别特征。注意：AudioSet 没有羽毛球类，零样本类别概率只能当弱特征用。

### 3.6 EfficientAT / BEATs / AST / YAMNet（备选嵌入）
- **EfficientAT**：https://github.com/fschmid56/EfficientAT ，**MIT**，权重在 GitHub Release；`mn04_as` 0.98M 参数，mAP 43.2；`mn10_as` 4.88M 参数，mAP 47.1。是片段级标注模型，适合在起音窗口上取嵌入。
- **BEATs**：https://github.com/microsoft/unilm/tree/master/beats ，仓库 **MIT**，权重在 OneDrive（README 说本项目按根目录 LICENSE 授权）。模型较大，不建议首选。
- **AST**：https://github.com/YuanGongND/ast ，**BSD-3**；AudioSet mAP 0.459（单模型，权重平均）/ 0.485（集成），权重在 Dropbox。模型大，时间分辨率粗。
- **YAMNet**：https://github.com/tensorflow/models/tree/master/research/audioset/yamnet ，代码 **Apache-2.0**（tensorflow/models 的 LICENSE 覆盖 `/research`）。权重 `yamnet.h5` 在 storage.googleapis.com，**权重单独的许可证未核实**（第三方转载页写 apache-2.0，不作为证据）。窗口 0.96 s、跳步 0.48 s，**时间太粗，不适合定位击球**，最多用来识别观众和说话声。

### 3.7 madmom / aubio / essentia（许可证陷阱）
- **madmom**：https://github.com/CPJKU/madmom 。LICENSE 文件写明：**源代码 BSD-2**，**所有模型和数据文件（.npz/.pkl 等）是 CC BY-NC-SA 4.0**；PyPI 分类器也标了"Free for non-commercial use"。它最强的 `CNNOnsetProcessor`、`RNNOnsetProcessor` 依赖的正是这些 NC 模型，**商用不能用**。纯 DSP 的 `SpectralOnsetProcessor`（superflux）只用代码，理论上可以用，但 librosa 已经覆盖了这部分。
- **aubio**：https://github.com/aubio/aubio ，**GPL-3.0**（GitHub 与 PyPI 一致）。闭源商用会传染，避开。
- **essentia**：https://github.com/MTG/essentia ，**AGPL-3.0**，服务端使用也触发开源义务，避开（MTG 另有商业授权，需单独谈）。
- **接入**：别引入。在 `pyproject.toml` 加一个 `audio = ["librosa>=0.10", "scipy>=1.11", "soundfile>=0.12"]` extra 就够了。

### 3.8 Baughman 等：Detection of Tennis Events from Acoustic Data（ACM MMSports 2019，IBM）
- **链接**：https://shiqiang.wang/papers/AB_MMSports2019.pdf ；Semantic Scholar：https://www.semanticscholar.org/paper/5757cc038c864dc0f2d676794faaa4ff8961a97f
- **许可证**：只有论文，**没发布代码和数据**。
- **精度**：数据来自 US Open Series 硬地职业赛（不同场馆、不同制作质量），共 6568 段 1 s 声音，约 1.83 h，4 折交叉验证。类别有 Announcer、Applause、Feet、Hit、Nonplay、Out。CNN + MFCC-Δ-ΔΔ 的 P/R/F1 为 92.47/92.40/**92.39%**（只用 MFCC 时 F1 89.88%，GMM 约 72–78%）。回合起点 P/R 84/82%，终点 77/77%。
- **接入**：给我们的第二级"起音分类器"提供设计和负类清单（脚步/鞋声、观众、解说/喊叫、出界）。窗口以峰值为中心，结构和 3.1 的起音栈正好衔接。

### 3.9 Huang & Cox：Improved Detection of Ball Hit Events in a Tennis Game Using Multimodal Information（AVSP 2011）及其前作
- **链接**：https://www.isca-archive.org/avsp_2011/huang11_avsp.pdf ；仓储记录 https://ueaeprints.uea.ac.uk/id/eprint/35438/ ；前作 "Detection of ball hits in a tennis game using audio and visual information"：https://ueaeprints.uea.ac.uk/id/eprint/42788/
- **许可证**：只有论文；没有代码和数据。
- **方法**（已读 ISCA PDF）：
  - 音频：30 ms 帧、10 ms 跳步，39 维 MFCC，每类一个 GMM（击球类 3 个分量）。置信度用"最高与次高类别对数似然之差"，以缓解信道失配。类别还包括观众、解说、裁判和司线员说话、蜂鸣、静音。
  - 视觉：先用单应做相机运动补偿，再做 blob 分类、tracklet 和图关联来跟踪球，球速度突变处记为击球。
  - 融合：分阶段的高层融合。以每个视觉击球时刻为中心建高斯窗，只接受窗内的音频击球；再用击球间隔分布作为时序约束。
- **精度**：数据是澳网 2008 女双 4 段（316/250/385/135 次，共 1086 次击球），音频模型用温网 2008 男单训练。F 值依次为：
  - 仅音频（似然比加时序）：75.54 / 77.30 / 73.47 / 76.23；
  - 仅视频：70.61 / 70.12 / 69.76 / 67.64（精度约 59–62%，召回约 78–82%）；
  - **融合：79.93 / 80.17 / 77.04 / 84.21**（精度 77–86%，召回 76–83%）。
  - 音频窗口取视觉击球 ±8 帧；文中把需要容差的原因归为视觉标注精度和"拍到麦克风约 30 ms 的声音延迟"。
- **接入**：这是和 RallyIQ 设置最接近的公开工作（单路音频加视觉球轨迹拐点），结论也和我们的预期一致：融合主要**提高精度**（视频单独时精度只有约 60%），召回变化不大。第 7 节第 4(a) 步的"视觉候选 ± 窗口内查音频起音"就是它的现代化版本：用学习打分器替代硬规则，用逐视频 Δ 和距离补偿替代固定的 ±8 帧宽窗。

### 3.10 Ebenezer 等：Detection of Audio-Video Synchronization Errors via Event Detection（ICASSP 2021，Amazon）
- **链接**：https://arxiv.org/abs/2104.10116 ；项目页 https://joshuaebenezer.github.io/publication/avsync
- **许可证**：arXiv 论文页标 CC BY-NC-SA 4.0（指论文本身）；**无代码、无数据**。
- **精度**：约 6 小时网球视频（约 50.4 万帧），人为加入不同步。视觉网络检测拍击球，音频网络检测击球声；音频里检到击球但邻近区间的视频里没有，就判为不同步。摘要只说"high accuracy"，没给数字。
- **接入**：正是我们要的**逐视频 Δ 自校准**。RallyIQ 的视觉击球检测器已经存在，只需加上音频起音和互相关（第 7 节第 2 步）。它的反向用法（"音频有、视频没有"）就是我们的"漏检找回"。

### 3.11 Synchformer / SparseSync（通用音画同步模型）
- **链接**：https://github.com/v-iashin/Synchformer （ICASSP 2024）；https://github.com/v-iashin/SparseSync
- **许可证**：两个仓库都是 **MIT**（Synchformer 的 LICENSE 已读）。checkpoint 托管在 a3s.fi，没有另写许可证。
- **精度**：偏移被当作分类问题，示例偏移 ±2 s。README 报告：LRS3 上 Acc@1 / Acc@1±1 类 = 86.6 / 99.6；VGGSound-Sparse 上为 43.8 / 60.2（VGGSound 训练）和 47.2 / 67.4（AudioSet 训练）。
- **接入**：**粒度太粗**（类间隔是 0.2 s 量级），不能替代事件互相关。只适合当"体检"：如果事件互相关失败，或者估出的偏移超过 ±200 ms，就用它确认是不是剪辑或转码导致了大幅错位。可选，不放进主路径。

### 3.12 TT Sounds：Sound-Based Spin Estimation in Table Tennis（Gossard 等，Tübingen，Sony AI 资助）
- **链接**：论文 https://arxiv.org/abs/2409.11760 ；数据和代码 https://github.com/cogsys-tuebingen/tt_sounds
- **许可证**：README 声明"All files in this dataset… **CC BY-NC 4.0**"；仓库**没有 LICENSE 文件**，代码也按非商用或保留权利处理。
- **精度**（已读 v2 正文）：3396 个拍面击球声（10 种球拍配置），另有 777 个球台、290 个地面、1239 个其他声音，按 80/20 划分。录音设备是 Zoom H4n Pro 指向性麦克风，44.1 kHz，**距离 0.5–2 m**。
  - 起音检测：1 ms 帧的平均能量与指数滑动平均比较，前面加 10 kHz 的 5 阶 Butterworth 零相位高通（乒乓球触拍能量集中在约 11 kHz）。无噪声时 **P/R 0.98/1.00、起音误差 0.09 ms**；叠加语音噪声时 P/R 0.98/0.95、起音误差 0.2 ms。
  - 分类：CNN 输入 15 ms（661 个采样点）、64 mel；旋转的分类准确率依次为下旋 87%、不转 96%、上旋 78%；推理 4.2 ms（i7 CPU）。
- **接入**：只作设计参考，不能进商用模型。它说明三点：
  - "零相位滤波 + 1 ms 能量帧 + 自适应基线"能把起音定位到亚毫秒；
  - 15 ms 的极短窗口已经够区分拍、台、地，对应羽毛球的"击球 / 球落地 / 其他"；
  - 但它的麦克风距离只有 0.5–2 m，我们是 3–17 m 加大馆混响和邻场干扰，精度会明显下降，不能直接类比。

### 3.13 Vanderplaetse & Dupont：Improved Soccer Action Spotting using both Audio and Video Streams（CVPRW 2020）
- **链接**：https://arxiv.org/abs/2011.04258 ；https://openaccess.thecvf.com/content_CVPRW_2020/html/w53/Vanderplaetse_Improved_Soccer_Action_Spotting_Using_Both_Audio_and_Video_Streams_CVPRW_2020_paper.html
- **许可证**：论文；摘要页没给代码链接。
- **精度**：SoccerNet（500 场），在纯视频架构里比较了不同阶段的音视频融合；加入音频后动作分类 mAP 平均绝对 **+7.43%**，动作定位 mAP **+4.19%**。
- **接入**：给"音频作为特征融合进现有打分器能带来多少增益"一个外部量级参考：几个点，不是翻倍。这也支持我们先做**特征级、后期融合**，不急着做端到端音视频网络。

### 3.14 其他相关（简述）
- **Hear the Sweet Spot**（Applied Sciences 2026，https://doi.org/10.3390/app16147340 ，文章 CC BY 3.0，无代码）：iPhone 15 Plus 采音，单通道定位网球拍面击球点。击球事件用"短时 RMS 能量峰加自适应阈值"离线提取（这一步没报精度），459 个样本，多专家 1D-CNN。它证明手机麦克风足以支撑击球音频分析，以后可扩展成教练报告里的"甜区击球率"。
- **Sharma 等，Wearable Audio and IMU Based Shot Detection in Racquet Sports**（https://arxiv.org/abs/1805.05456 ）：乒乓球，腕戴音频加 IMU 融合，准确率 95.6%；用检测到的击球事件对齐两路传感器，和 3.10 是同一个思路。没有代码。
- **Hajdu-Szücs 等，Audio-based performance evaluation of squash players**（PLoS ONE 2018，https://arxiv.org/abs/1704.08765 ）：壁球场内用声音判断事件类型并做定位。摘要没有给传声器配置和精度，许可证未核实。多麦定位不适合单手机，只作背景参考。
- **pyroomacoustics**（https://github.com/LCAV/pyroomacoustics ，**MIT**）：有 GCC-PHAT 和 DOA 实现，可以从双声道录音里估计 ITD（用于串音特征）；也能生成房间冲激响应做混响数据增强。
- **DESED_task / sed_eval / psds_eval**（DCASE 基线 https://github.com/DCASE-REPO/DESED_task ，**MIT**；sed_eval、psds_eval 在 PyPI 上均为 MIT）：事件级 P/R/F1（带 collar 容差）与 PSDS 评测，可直接拿来评估音频流。
- **torchaudio**（BSD-2）：README 写明已进入**维护期**，2.9 起删掉了部分功能（含 I/O）。音频解码继续用 ffmpeg 加 soundfile，不要依赖 torchaudio 的 I/O。
- 背景证据（不是音频）：*Can Event Detectors Leave the Broadcast Camera? Evidence from Tennis*（arXiv 2610.38042）发现，转播权重的 E2E-Spot 在手机和场边机位上弹地 F1 只有 0.130；无训练的仲裁层在击球**时刻**上离接触帧太远；零样本 VLM 平均差约 0.1 s。这正是音频起音能补的短板：接触时刻的精确定位。

---

## 4. 多场地串音（cross-talk）怎么处理

**问题规模（估算）：** 设可听见的邻场有 3–5 片，每片回合期间约每 1–1.2 s 一拍、约一半时间在打，合计 λ ≈ 1.5–2.5 个外来起音每秒。匹配窗口宽 w = 50 ms（±25 ms）时，碰巧命中的概率 p ≈ 1 − e^(−λw) ≈ 7–12%。底线后机位下，邻场的近端击球可能比本场的远端击球**更响**，所以**单靠音量没法区分**。

按收益和成本排序的手段：

1. **时间门控（必须）**：只在视觉回合区间（`rallies.py`）前后各延 0.5 s 内用音频。音频起音必须落在"视觉候选时间 + 声速延迟 + Δ"的 ±max(25 ms, 1 帧) 内，才算确认。
2. **逐视频"本场声学模型"（推荐，几乎零成本）**：
   - 正样本：和视觉高置信击球（打分器置信度 > 0.8，`track_quality` 高）在 ±20 ms 内匹配上的起音；
   - 负样本：(a) **本场回合间歇**里的所有起音（这时本场没人击球，但邻场还在打）；(b) 本场回合内、球在高置信平滑飞行段（没有拐点、离球员远）时出现的起音；
   - 特征：**按预测距离校正后的响度** `peak_dB + 20·log10(d_pred)`、SNR、频谱质心、1–3/3–6/6–12 kHz 能量比、早晚能量比（起音后 0–10 ms 对 10–80 ms，粗略反映直达声和混响之比）、双声道 ITD/ILD（有立体声时）；
   - 模型：带 L2 正则的逻辑回归，在每段视频上拟合。样本不够（正样本少于 15 个）时退回全局先验模型。输出 `aud_court_p` 作为打分特征。
3. **双声道方位（可选特征，不当门控）**：手机横拿时两个麦克风相距约十几厘米，最大 ITD 约 0.4 ms（48 kHz 下约 20 个采样点）。可以用 GCC-PHAT 在 20 ms 窗口内估计。但底线后机位时，本场近端击球的方位角也可能到 ±40°，和邻场有重叠；混响和机内降噪也会削弱这条线索。所以只作特征。很多设备录单声道，或者两声道其实一样；要先检测 L−R 相关性，再决定用不用。
4. **节奏先验**：本场相邻两拍间隔约 0.25–2.0 s，并且击球方交替。120 ms 内出现两个起音时只留与轨迹一致的那个；音频提出的新候选也必须满足交替规则（`R_SAME_HITTER` 逻辑可以复用）。
5. **纯音频候选的额外门槛（多场馆必需）**：在起音时刻 ±150 ms 内必须**同时**满足：某个本场球员手腕或预测拍头速度有峰值（RTMO 姿态），并且该时刻附近的球轨迹出现缺口或者方向翻转。否则只标 `needs_review`，不计入统计。

---

## 5. 手机 / 运动相机的音画偏移

| 来源 | 量级 | 处理方式 |
|---|---|---|
| 容器层：音视频流 `start_time` 不同、MP4 edit list | 常见几十 ms，个别情况更大 | `ffprobe -show_entries stream=index,codec_type,start_time,time_base` 读出初值 Δ₀；用 ffmpeg 默认会应用 edit list 的方式解码，保证两路走同一时间轴 |
| 手机可变帧率（VFR） | 按平均帧率换算会累积"几帧"误差（RallyIQ 的 `quality.py` 已经提示） | VFR 视频读取逐帧 PTS（`ffprobe -select_streams v:0 -show_entries packet=pts_time`），建一张"帧号 → 真实时间"表，音频匹配一律用真实 PTS |
| 设备采集和编码管线延迟 | 因机型而异；论坛普遍反映 GoPro 等运动相机会差"几帧"（只是轶事，未量化） | 不做假设，交给第 2 步的事件互相关逐视频估计 |
| 声速传播 | 9–50 ms（见第 1 节） | 每次击球按 `d / c` 补偿；d 用标定反推的相机地面位置到击球者脚点（`foot_court_xy`）的三维距离，击球点高度按 2 m 粗估 |
| 时钟漂移 | 长视频里可能出现线性漂移 | 用 120 s 滑窗分段估 Δ(t)，再做线性拟合；残差变大时按段处理 |

**人能察觉的阈值**（参考，转引自 Wikipedia "Audio-to-video synchronization" 条目）：
- ITU-R BT.1359-1 的可察觉范围是**声音提前 45 ms 到滞后 125 ms**；
- EBU R37 要求端到端控制在 +40 / −60 ms 以内；
- ATSC IS-191 要求声音提前不超过 15 ms、滞后不超过 45 ms。

这意味着设备厂商只需保证几十毫秒的同步，人就看不出来。但**我们对算法的要求比这严格得多**：需要 ≤ 半帧（约 8–16 ms），所以必须用事件级估计。

---

## 6. 许可证友好库清单

| 用途 | 推荐 | 许可证 | 备注 |
|---|---|---|---|
| 解码 | ffmpeg CLI（已在用）+ soundfile | FFmpeg 按构建是 LGPL 或 GPL；soundfile BSD-3 | 以子进程调用 CLI，不链接 |
| DSP 起音 | **librosa**、**scipy** | ISC / BSD | SuperFlux、`sosfiltfilt`、`find_peaks` |
| 帧级 SED 骨干 | **PretrainedSED**（frame_mn06/mn10） | MIT（代码 + Release 权重） | 首选可微调模型 |
| 嵌入和零样本类别 | **PANNs** / panns_inference | 代码 MIT；权重 **CC BY 4.0**（需署名） | 有帧级输出 |
| 轻量嵌入 | EfficientAT | MIT | mn04 不到 1M 参数 |
| 大模型嵌入 | BEATs（unilm）、AST | MIT / BSD-3 | 慢，不首选 |
| 音画粗同步 | Synchformer | MIT | 只用于体检 |
| 立体声 ITD、RIR 增强 | pyroomacoustics | MIT | |
| 评测 | sed_eval / psds_eval / DESED_task | MIT | |
| **避开** | madmom 模型文件 | **CC BY-NC-SA 4.0** | 代码 BSD，模型不可商用 |
| **避开** | aubio / essentia | **GPL-3 / AGPL-3** | |
| **避开（数据）** | TT Sounds | **CC BY-NC 4.0** | 只作研究参考 |

**关于 AudioSet 派生权重的提醒：** PANNs、PretrainedSED、EfficientAT、BEATs、AST 都在 AudioSet 上预训练。AudioSet 发布的是 YouTube ID 加标签，音频本身归上传者所有。权重的代码许可证虽然宽松，但"用受版权保护的音频训练出的权重能否商用"在法律上并不完全明确。稳妥做法：第一版只用 DSP 起音栈（不含任何预训练权重）加逐视频逻辑回归；如果要用 AudioSet 预训练的骨干，先过一次法务评审，并在 NOTICE 里给 PANNs 权重署名（CC BY 4.0 要求）。同理，ShuttleSet 和 BWF 转播的音频（`scripts/public_eval.py` 下载时用了 `+ba`）只能用于内部评测，不宜用来训练商用模型。

---

## 7. 落地配方：给 RallyIQ 击球检测器加"音频候选流"

### 第 0 步：抽取音频（新 stage，约 0.5 天）
- `rallyiq/pipeline/probe.py`：在现有的 `v:0` 探测之外，再探测音频流（`has_audio`、`sample_rate`、`channels`、`start_time`），写进 `VideoMeta`（新增可选字段）。VFR 视频额外导出逐帧 PTS 表。
- 新 stage `audio`（注册到 `rallyiq/pipeline/registry.py`，放在 `probe` 之后）：
  `ffmpeg -v error -i IN -map 0:a:0 -vn -ac 2 -ar 48000 -c:a pcm_s16le audio.wav`（保留双声道，同时生成单声道混音）。
- `rallyiq/ingest/quality.py` 加音频质量项：没有音轨、单声道或伪立体声（L/R 相关性 > 0.99）、削波比例、风噪滤波或 AGC 的明显迹象（底噪"抽吸"）。

### 第 1 步：DSP 起音（新模块 `rallyiq/audio/onsets.py`，约 1 天）
1. 带通：`butter(4, [1500, 10000], btype="band", fs=48000, output="sos")` 加 `sosfiltfilt`（零相位）。上限先取 10 kHz，之后用已确认击球做逐频带 SNR 统计再调。
2. 两路包络：RMS（5 ms 窗、1 ms 跳）；SuperFlux：`librosa.onset.onset_strength(y, sr=48000, hop_length=240, lag=2, max_size=3, fmin=1000, n_mels=128)`。两路归一化后取几何平均。
3. 自适应阈值：4 s 滑窗"中位数 + k·MAD"，k 初值 5，并要求 SNR ≥ 6 dB；不应期 120 ms。
4. 起音精修：在峰前 30 ms 内找带通包络第一次超过"底噪 + 0.3 ×（峰值 − 底噪）"的采样点，精度到亚毫秒。
5. 每个起音输出：`t_ms`、`peak_db`、`snr_db`、上升时间、衰减时间、频谱质心、三个频带能量比、频谱平坦度、早晚能量比、`itd_ms` 和 `ild_db`（双声道时用 GCC-PHAT，取 ±0.6 ms 搜索范围）。产物写成 `audio_onsets.json`（新 contract）。

### 第 2 步：逐视频 Δ 自校准（`rallyiq/audio/sync.py`，约 1 天）
- 输入：视觉高置信击球的时间 `t_v`（用真实 PTS）及其距离 `d_i`。
- 搜索：Δ 在 [−500, +500] ms 内按 1 ms 步长，得分 `S(Δ) = Σ_i max_j exp(−(t_a_j − (t_v_i + d_i/c + Δ))² / (2σ²))`，σ = 半帧。取 argmax，再用 ±20 ms 内匹配对的残差中位数精修。
- 置信条件：至少 8 对匹配，主峰和次峰得分比大于 1.5，残差 MAD ≤ 10 ms。不满足就退回容器的 Δ₀，并把 `audio_calibrated` 设为 false（后面的音频特征降权或置空）。
- 长视频用 120 s 滑窗估 Δ(t)，检查漂移。
- 说明：视觉拐点相对真实接触的系统偏差会一起被吸收进 Δ，这正是我们想要的。

### 第 3 步：本场声学模型（`rallyiq/audio/crosstalk.py`，约 1 天）
按第 4 节第 2 点构造正负样本和特征，拟合逐视频逻辑回归，给每个起音打 `aud_court_p`。没有回合间歇（短片段）时用全局先验模型。

### 第 4 步：融合进 `rallyiq/events/hits.py` 与 `learned.py`（约 2 天）
- **(a) 特征融合（最小改动、最安全）**：在 `detect_hits()` 的 NMS 之后、打分之前，给每个 `Candidate` 查最近的起音，新增 `learned.FEATURES`：
  `aud_ok`（这段视频有音频且已校准）、`aud_match`（±max(25 ms, 1 帧) 内有起音）、`aud_dt`（残差，以帧为单位，截断到 ±3）、`aud_snr`、`aud_court_p`，可选再加 `aud_pann_hit`。
  训练时对这些特征做 **30–50% 随机置空**（缺失时 `aud_ok = 0`），这样同一个模型在没有音频或音频没校准时也能用。ShuttleSet 转播音频有解说、观众声和制作延迟，和业余音频分布不同：在 ShuttleSet 上只学"有没有音频"的结构，权重要用业余标注做微调或校准。
- **(b) 纯音频候选（找回漏检）**：对 `aud_court_p > 0.6`、并且 ±80 ms 内没有视觉候选的起音，在 `t_a − d/c − Δ` 对应的帧建一个新 `Candidate`，原因码 `R_AUDIO_ONLY = "audio_only_candidate"`；然后照常算 `poses.proximities()`。只有满足第 4 节第 5 点（姿态峰值加轨迹缺口或翻转）才放进 NMS，否则直接 `needs_review`。距离 d 用候选时刻最近球员的脚点来估。
- **(c) 时间精修**：已确认的击球把帧号吸附到 `t_a − d/c − Δ` 对应的帧（或保留亚帧时间）。下游的球速、击球高度和挥拍阶段特征（stroke 模型）都会受益。
- 打分器选择逻辑沿用 `EventsConfig.hit_scorer`；新增配置 `audio_enabled`、`audio_band_hz`、`audio_k_mad`、`audio_refractory_ms`、`audio_match_ms`、`speed_of_sound_mps`。

### 第 5 步（可选）：学习型音频头（约 1 周）
- 用第 2、3 步的自监督样本，加少量人工复核，微调 PretrainedSED `frame_mn06`，或者训练一个小 CRNN（128 mel、5 ms 跳步）。输入是以起音为中心的 0.5 s 片段；输出两类（本场击球 / 其他），辅助类可加球落地、鞋声、人声。
- 数据增强：混入**其他视频回合间歇的音频**（真实多场馆背景）；随机增益 ±12 dB；pyroomacoustics 生成的 RIR 混响；AAC 重编码。
- 先做 3.1–3.5 的无训练版本，只有当第 7 节实验显示瓶颈在"起音分类"而不是"起音漏检"时才做这一步。

### 评测与 go/no-go（先做，1 天）
1. 拿 DJI_0009（58 个人工标注击球）跑第 0–2 步，报告：Δ 是否收敛；起音对 58 个击球的召回（±25 ms 和 ±1 帧两档，按近端/远端、杀球/高远/网前分组）；回合内多余起音的数量。
2. 再标注 3–5 段**多场馆**业余视频（含双打），重复上面的统计，并对比第 3 步本场模型启用前后的精度。
3. 指标：事件级 P/R/F1（±1 帧、±2 帧两档），命中击球的时间误差中位数；用 sed_eval 或现有的 `evaluation/metrics.py`。
4. Go 条件：Δ 在 ≥ 90% 的视频上收敛且残差 MAD ≤ 10 ms；单场地本场击球的起音召回近端 ≥ 85%、远端 ≥ 70%；接入第 4(a)(b) 步后，击球 F1 提升 ≥ 5 个点，且精度下降 ≤ 2 个点。

---

## 8. 预期收益（估算，请用第 7 节实验验证）

- **起音召回（单场地）**：近端的高远球、杀球、平抽这类重击，接近全部；远端重击大部分能抓到；**网前搓、放、挡这类轻击**在远端可能只有一半上下，而这些恰好也是视觉的难点，所以不能指望音频补上它们。
- **外部量级参考**：
  - 网球转播（Huang & Cox 2011，3.9）：音频加视觉融合比仅视觉高 **7–17 个 F 点**（67.6–70.6 → 77.0–84.2），比仅音频高 2–8 个点，增益主要来自**精度**；
  - 足球（3.13）：加音频后动作定位 +4.19 mAP。
  - RallyIQ 的视觉基线比 2011 年的强得多，增益应该介于两者之间。
- **击球召回**：视觉漏检主要来自遮挡、运动模糊和低机位的拐点不明显；音频漏检主要来自轻击和掩蔽，两者有较大互补性。DJI_0009 上规则版能找到 47/58，学习打分器只有 34/58，说明很多真击球**已经是候选，只是被打分器拒绝了**。带有"匹配起音"特征的打分器最容易把它们捞回来。粗估：
  - 学习打分器加音频特征和纯音频候选：34/58 可能到 **48–53/58**；
  - 规则版：47/58 可能到 50–54/58；
  - 多场馆要打折扣，纯音频候选的贡献会因第 4 节第 5 点的门槛而减少一部分。
  - 对我们帮助最大的应该是"确认被打分器错拒的候选"和"时间精修"，其次才是"找回视觉完全漏掉的击球"。
- **时间精度**：命中击球的时间误差中位数从约 1 帧降到半帧以内（音频起音本身是毫秒级，残余误差主要来自 Δ 和距离估计）。这对后续的球速、击球高度和挥拍阶段特征都有利。
- **成本**：第 0–4 步约 1–1.5 人周，CPU 开销可以忽略（DSP 每段视频几秒）；第 5 步另需约 1 人周和少量 GPU。

---

## 9. 未核实 / 风险清单

- YAMNet 权重文件单独的许可证没在一手来源上核实。
- Hajdu-Szücs 等（壁球）的许可证和精度没有核实；Baughman 等的 PDF 只读到了方法和结果摘要。
- 业余羽毛球音频仓库（duzitong、riven314）都没有 LICENSE 文件，**不能复用代码**，只作可行性参考。
- 运动相机和手机的典型 A/V 偏移量没有找到系统测量，只有论坛轶事，所以配方采用逐视频估计、不做假设。
- 立体声 ITD 在大馆混响和机内降噪下是否可用，需要用实测确认，所以只作特征。
- 设备侧的"风噪抑制、语音增强"类处理可能压低击球瞬态（DJI 和部分手机默认开启）。在 `quality.py` 里检测，并建议用户关闭。
- 第 8 节的所有收益数字都是**估算**，没有公开的业余羽毛球音频基准可以引用。
