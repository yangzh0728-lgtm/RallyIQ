# RallyIQ 调研：羽毛球/小球追踪 · 击球帧精确定位 · 广播→业余机位迁移（2023–2026 新工作）

> 注：本文件是原始调研稿。许可证和数字已另行逐条核查，更正见上级目录 `2026-11-28-web-research.md` 第三节和 `verification.json`；两者不一致时以核查结果为准。


> 调研日期：2026-11-28。已排除需求中列出的已知工作（TrackNet V1–V4、MonoTrack、2307.16000、AICUP DSNet、SwingNet、BadmintonAI、shuttle-speed-cam、MatchClip-AI、shuttlecut）。
>
> **核实方式**：许可证用 GitHub search API 的 `license.spdx_id` 字段核对，并抓取 `raw.githubusercontent.com` 上的 README / LICENSE 文件交叉确认。"none" 表示仓库中没有 LICENSE 文件，按默认规则为"保留所有权利"，不能直接商用。指标来自论文正文或 README。HuggingFace 模型卡的许可证无法从本环境访问，凡涉及处均注明"未核实"。

---

## 0. TL;DR：Top 5 可落地项（按对 RallyIQ 瓶颈的性价比排序）

| # | 工作 | 打哪个瓶颈 | 代码 / 权重许可 | 工作量 | 一句话理由 |
|---|---|---|---|---|---|
| 1 | **RacketVision**（AAAI 2026 Oral） | 侧机位误检（球拍、手臂）+ 击球判定 | 代码 MIT，数据集 MIT（HF 标签），权重在 HF（模型卡许可未核实） | 低–中（3–5 天） | 唯一公开的**羽毛球球拍 5 点姿态**模型和数据。可用拍头位置做击球特征，也能屏蔽拍面上的误检。同一基准还证明**中值背景 + 多帧输入**让 TrackNetV3 在羽毛球上的 mAP@50 从 57.8 升到 83.1。 |
| 2 | **AdaSpot**（CVPR 2026），基于 **E2E-Spot** 代码框架 | 业余机位击球召回（34/58） | AdaSpot 为 MIT，权重在 HF；E2E-Spot 代码 BSD-3 | 中（1–2 周加 GPU） | 端到端 RGB 精确事件定位的当前 SOTA，对每帧中最显著的区域做高分辨率处理，适合广角业余画面里的细小拍–球接触。先在 ShuttleSet 击球帧上预训练，再用几十个业余回合微调。 |
| 3 | **UMEG-Net**（AAAI 2026） | 小样本、跨机位的击球检测 | 无 LICENSE，无权重，需**按论文复现** | 中（1–2 周） | 把"姿态 + 球点 + 球场角点"建成统一图，用 GCN 加多尺度时间移位，专为小样本设计。在 ShuttleSet 上只用 100 个 clip 训练，事件 F1 为 59.1，E2E-Spot 为 42.7，T-DEED 为 33.8。RallyIQ 已有 RTMO 和 TrackNet 的输出，可以直接喂入。 |
| 4 | **WASB**（BMVC 2023）+ **BlurBall**（CVPRW 2026） | 羽毛球误检和定位精度 | 均为 MIT；WASB 提供羽毛球权重，BlurBall 提供乒乓球权重 | 低–中 | 加一个 HRNet 检测器（1.5M 参数）与 TrackNetV3 做一致性投票，可以过滤误检，也能用来生成伪标签。BlurBall 的"模糊中心标注"在所有检测器上都有提升，比如 TrackNetV3 的 F1 从 93.7 升到 95.9。它的模糊方向输出能提供单帧运动方向，可作为击球线索。 |
| 5 | **kwyoke/Badminton-hit-detection** 业余数据集（2022） | 缺少业余域的评测和训练数据 | 无 LICENSE，**仅限内部评测** | 低（1–2 天） | 公开数据中少见的**业余单打和双打、底线机位、带击球帧标注**的集合。方法是 14 帧坐标序列（球场、姿态、球）加 GRU，作者称从职业比赛训练后能迁移到业余。可作为 58 击测试集之外的第二测试集。 |

**建议的两周路线**：
1. 第 1–2 天：用 kwyoke 业余集加自有视频建立固定评测，指标为召回、精度和 ±2 帧准确率。
2. 第 2–5 天：接入 RacketVision 的拍头关键点，并按 Hsu 2024 的规则加"挥拍门控"（见 §2.8）。确认 TrackNetV3 微调时启用了 `bg_mode`。
3. 第二周：训练 AdaSpot 或 E2E-Spot 的"hit"单类模型，或复现 UMEG-Net 的轻量图模型。产出每帧的击球概率，作为现有 GBM 打分器的新特征或第二候选源。

---

## 1. 小而快的球 / 羽毛球追踪

### 1.1 RacketVision：多球拍运动的球与球拍统一基准 ★Top1
- **链接**：论文 https://arxiv.org/abs/2511.17045（AAAI 2026 Oral）。代码 https://github.com/OrcustD/RacketVision 。数据 https://huggingface.co/datasets/linfeng302/RacketVision 。权重 https://huggingface.co/linfeng302/RacketVision-Models
- **许可**：代码 MIT，已核实 LICENSE。数据集在 HF 页面标注 MIT。权重通过 `download_checkpoints.py` 从 HF 下载，HF 模型卡许可**未核实**。
- **内容**：覆盖乒乓球、网球、羽毛球三项，共 1,672 个 clip、43.5 万帧，均为 YouTube 职业转播画面。标注包括球点和可见性，以及**球拍框和 5 个关键点**（top、bottom、handle、left、right）。仓库提供三个模块：BallTrack（基于 TrackNetV3）、RacketPose（RTMDet-M 检测加 RTMPose-M 5 点姿态）、TrajPred（结合球拍信息的轨迹预测）。
- **指标（羽毛球）**：
  - TrackNetV3 单帧、无背景：P 0.876，R 0.706，MDE 7.39px，mAP@50 57.8。
  - 加中值背景（BM）：P 0.924，R 0.797，MDE 3.34，mAP@50 67.7。
  - 三项运动联合训练 + BM + 4 帧：MDE 1.54，**mAP@50 83.1**。
  - 球拍姿态 PCK@0.2：羽毛球约 89%，乒乓球约 82%，网球约 90%。
  - 轨迹预测中，简单拼接球拍特征反而变差，用 cross-attention 融合才有小幅提升。
- **接入 RallyIQ**：
  1. 在 RTMO 人体框附近跑 RacketPose。用拍头关键点的速度、拍面角速度以及"拍头到球的距离"作为击球打分特征。这些特征几乎与机位无关，可替代不能迁移的广播方向特征。
  2. 落在球拍框或手臂区域内的球候选降权，直接针对侧机位的拍面误检。
  3. 确认 TrackNetV3 微调时使用 `bg_mode`（中值背景）。业余固定三脚架的中值背景比转播更干净，能明显压制静止白色物体（灯、线、场边杂物）。
  - **注意**：球拍模型只在转播画面上训练过，需要在自家低机位素材上抽检 PCK，必要时补标 200–500 帧再微调。

### 1.2 WASB：Widely Applicable Strong Baseline for Sports Ball Detection & Tracking ★Top4
- **链接**：https://arxiv.org/abs/2311.05237（BMVC 2023）。代码 https://github.com/nttcom/WASB-SBDT ，权重列表见 MODEL_ZOO.md。
- **许可**：代码 MIT（已核实 LICENSE.md，NTT Communications 2023）。MODEL_ZOO 提供 5 个运动、7 种模型（DeepBall、BallSeg、TrackNetV2、MonoTrack、WASB 等）的 Google Drive 权重，**含羽毛球**。权重没有单独声明许可，默认随仓库 MIT，但训练数据有各自条款。
- **方法**：HRNet-small 去掉 stem 下采样以保持高分辨率特征，3 帧输入、3 帧输出，约 1.5M 参数。训练时对难样本使用"位置感知"的实值热图加 quality focal loss。推理时用局部运动模型做时间一致性候选选择。
- **指标（羽毛球，τ=4px）**：WASB step=1 时 F1 93.1，Acc 89.0，AP 91.6。对比 TrackNetV2 的 F1 90.5 和 MonoTrack 的 90.9。论文没有与 TrackNetV3 直接比较；RacketVision 的羽毛球表中，WASB 加 BM 的 MDE 为 4.03，TrackNetV3 加 BM 为 3.34，两者接近。
- **接入 RallyIQ**：作为第二检测器与 TrackNetV3 并行。两者在同一位置都高置信的点作为伪标签，用于在自有业余素材上自训练；两者分歧的帧标记为不可信，可降低白色物体和球拍误检。工作量低（官方提供 Docker），推理成本小。

### 1.3 BlurBall：联合估计球位置与运动模糊 ★Top4
- **链接**：https://arxiv.org/abs/2509.18387（CVPRW 2026 CVsports）。代码 https://github.com/cogsys-tuebingen/blurball 。项目页 https://cogsys-tuebingen.github.io/blurball
- **许可**：代码 MIT（GitHub 检测）。权重（BlurBall 以及 WASB、TrackNetV2、MonoTrack 等基线）在图宾根大学 Nextcloud，README 已给链接。数据集（6.4 万帧乒乓球，含业余和职业比赛）同样托管在 Nextcloud，**数据许可未声明**。
- **方法与指标**：
  - 基于 WASB 代码，加入 SE 注意力，同时预测球心、模糊方向和模糊长度。
  - **"模糊中心标注"**指把标注点放在拖影中点而不是前沿。这一改动让所有检测器都有提升，例如 TrackNetV3 的 F1 从 93.74 升到 95.93，WASB 从 95.58 升到 96.00。
  - BlurBall 的 F1 为 97.17，Acc 94.75；模糊长度和角度误差约减半。
  - 位置加模糊信息做轨迹拟合时，MAE 从 84.4px 降到 53.0px。
- **接入 RallyIQ**：
  1. 30fps 的业余素材里，羽毛球几乎每帧都是拖影。**自有微调数据统一按拖影中点标注**，成本为零，收益已有验证。
  2. 中期可以在羽毛球数据上训练模糊方向头。单帧就能给出运动方向，击球前后拖影方向的突变可以作为不依赖机位标定的击球线索。这需要额外标注模糊，工作量中等。

### 1.4 TOTNet：遮挡感知的时序球追踪
- **链接**：https://arxiv.org/abs/2508.09650（已发表于 CVIU 2026）。代码 https://github.com/AugustRushG/TOTNet
- **许可**：代码 MIT。**没有发布预训练权重**，README 只说明训练会保存 checkpoint。附带的 TTA 残奥乒乓球遮挡数据集仅限研究用途，需签署协议后在 HF 申请。
- **方法与指标**：
  - 结构为保留时间维的 3D 卷积 U-Net，配合可见性加权 BCE、**遮挡增强（随机遮掉球）**，可选 RAFT 光流。
  - 在 TTA 完全遮挡帧上，准确率从 WASB 的 0.63 升到 0.80，RMSE 从 37.3 降到 7.19。
  - 在羽毛球上与 WASB 大体持平：可见帧更好，不可见帧更差。
  - README 指出输入超过 5 帧不再提升，原因是球拍运动中击球会使方向突变。
- **接入 RallyIQ**：不建议替换主干。建议把**遮挡增强（遮掉球或用人体、球拍块覆盖）**和**按可见性加权的损失**移植到 TrackNetV3 微调中，专门对付侧机位下手臂和身体的遮挡。工作量低（约 1 天）。

### 1.5 TrackNetV5（两个同名、不同团队的工作）
- **(a) NYCU 版**：Chang、Wang、Huang，*Robust Shuttlecock Tracking via Motion Prompts and Spatiotemporal Attentive Fusion*，**ICIP 2026**。论文 PDF：https://people.cs.nycu.edu.tw/~yushuen/data/TracknetV5_26.pdf 。代码 https://github.com/gino6178/TrackNetV5
  - **许可**：仓库**没有 LICENSE**，因此不能商用。`releases/` 下有 Lite（α=0.5）和 Full 两个权重，README 说明它们是复现训练得到的。
  - **指标**：在 TrackNetV2/V3 的羽毛球数据集上，F1 为 98.47，对比 V3 的 97.64 和 V4 的 97.99。V5-Lite 只有 2.58M 参数，135 FPS，F1 98.06。
  - **方法**：R(2+1)D、门控残差运动提示、SCG 多尺度融合。它也把每个回合的中值背景作为第 4 通道（`bg_mode=subtract_concat`）。
- **(b) 武汉理工 + 上海 Code Zero 版**：https://arxiv.org/abs/2512.02789 。SDK 在 https://github.com/thaonan/TrackNetV5-SDK
  - **许可**：README 声明为**专有软件**，权重和数据不公开，代码仅供学术交流。
  - **指标**：在 TrackNetV2 数据集上 F1 0.9859，V4 为 0.9581。
  - **可借鉴点**：MDD 把帧差拆成正、负两个极性通道，保留运动方向。
- **接入 RallyIQ**：两者都只在转播数据上比较，相对 V3 提升不到 1 个点，也没有针对业余或遮挡的评测，**优先级低**。可以低成本借鉴 MDD 的思路：给 TrackNetV3 输入加两个正、负帧差通道，再在自有数据上做 A/B 测试。

### 1.6 One-Shot Badminton Shuttle Detection for Mobile Robots（ETH Zürich, 2026）
- **链接**：https://arxiv.org/abs/2603.06691 。代码 https://github.com/leggedrobotics/shuttle_detection
- **许可**：**AGPL-3.0**，属强 copyleft 且包含网络服务条款，只能作为内部标注工具使用。README 中的数据集链接还是占位符 example.com，**数据实际未公开**；权重未提及。
- **方法与指标**：单帧 YOLOv8，每帧最多保留一个检测。训练数据为 11 种背景、2 万帧，采用**半自动标注**：GMM 背景减除、YOLOv8-seg 去除人物、时间滤波，然后导入 CVAT 人工修正。在相似环境中 F1 为 0.86，在全新环境中为 0.70。
- **接入 RallyIQ**：价值在于**标注流水线**而不是模型。参考这套思路，为自有业余视频批量生成羽毛球候选标签，再人工修正，能大幅降低微调数据的成本。由于 AGPL，建议自己重写这几步（都是 OpenCV 和 Ultralytics 的常规操作）。

### 1.7 单目 3D 轨迹重建（新）
- **Where Is The Ball**（CVPRW 2025）：https://arxiv.org/abs/2506.05763 ，项目页 https://where-is-the-ball.github.io
  - **状态**：代码"承诺发布"，GitHub 上只找到项目页仓库，**未找到代码和权重**，许可未知。
  - **方法**：把每个 2D 点反投影成射线，再用射线与地面、竖直平面的交点表示，得到**与相机无关的规范表示**。三个 LSTM 分别判断轨迹结束、估计高度和做精修。**只用 Unity PhysX 合成数据训练**，就能泛化到真实网球数据：落点误差 0.63m，SynthNet 为 3.58m。
  - **对 RallyIQ 的关键启发**：这正是解决"广播训练的方向特征不能迁移"的一类做法。把轨迹特征从图像坐标换到**球场平面（或射线规范）坐标**后，打分器就不再依赖机位。详见 §3。
- **Uplifting Table Tennis**（WACV 2026）：https://arxiv.org/abs/2511.20250 ，代码 https://github.com/KieDani/UpliftingTableTennis
  - **许可**：GPL-3.0。
  - **内容**：用 MuJoCo 合成的物理轨迹训练 2D→3D 提升模型，用于真实乒乓球的 3D 轨迹和旋转估计；同时提供球台关键点检测和相机标定教程。
  - **对 RallyIQ**：可作为"合成轨迹 + 随机机位投影"训练流程的参考实现。羽毛球需要换成带空气阻力的羽毛球动力学模型。GPL 限制下只适合参考，不要并入产品。
- **FreeTrack**：https://github.com/Alexanderyk6/FreeTrack ，论文投稿 The Visual Computer，尚未确认接收。
  - **许可**：无 LICENSE，`weights/` 目录存在，README 未报告数值指标。
  - **方法**：YOLOv12 与 TrackNet 混合检测羽毛球；用 **XGBoost 在运动学特征上判定击球帧**，思路与 RallyIQ 相同；用 6 点标定（4 个场角加 2 个网柱）；假设每段飞行位于竖直"运动平面"内，逐帧做射线–平面求交得到 3D。
  - **对 RallyIQ**：6 点标定适合野外视频，运动平面法能低成本得到击球高度和落点；它的击球特征定义也可以对照参考。

### 1.8 事件相机（仅备注）
- *1000 Rallies: An Event-Camera Dataset and Real-Time Learned Ball-State Estimation for Robotic Table Tennis*（arXiv 2606.25620）。事件相机不适用于手机或 DJI 素材，**未深入核实代码**，不推荐。

---

## 2. 击球帧 / 精确事件定位（PES）

### 2.1 AdaSpot（CVPR 2026）+ E2E-Spot（基础框架） ★Top2
- **AdaSpot**：https://arxiv.org/abs/2602.22073 。代码 https://github.com/arturxe2/AdaSpot ，权重 https://huggingface.co/arturxe/AdaSpot
  - **许可**：代码 MIT。HF 权重许可未核实。
  - **方法**：低分辨率全帧提取全局特征，再用**无需训练的显著性 RoI 选择器**，为每帧选一个区域做高分辨率处理，最后接 BiGRU。骨干为 RegNetY-200MF 或 400MF。
  - **指标**：在严格指标下为 SOTA，Tennis mAP@0 帧比此前最好方法高 **3.96**，FineDiving 高 2.26。参考同表基线：E2E-Spot 200MF 的 Tennis mAP@δ=0/1/2 为 69.8/97.0/97.7。
  - 配置支持 Tennis、FineDiving、FineGym、**F3Set**、SoccerNetBall。
- **E2E-Spot**（ECCV 2022，作为基础框架列出）：https://github.com/jhong93/spot
  - **许可**：代码 BSD-3-Clause。权重仓库 https://github.com/jhong93/e2e-spot-models **没有 LICENSE**，但 RallyIQ 本来就需要自训。
  - **自定义数据格式**：`{split}.json` 中写 `video/num_frames/fps/events[{frame,label}]`，帧图缩放到约 224px 高。
- **接入 RallyIQ**：
  - 以"hit"单类，或"hit_near / hit_far"两类做训练。先用 ShuttleSet 的击球帧预训练，再用 30–100 个自有业余回合微调。
  - 把每帧的击球概率作为 GBM 打分器的新特征，或作为与轨迹拐点并行的第二候选源。
  - **为什么选 AdaSpot**：业余广角画面缩到 224px 高后，球和拍几乎消失，而 AdaSpot 的高分辨率 RoI 正好补上这一点。
  - **证据**：§3 的 Beyond the Premier 显示端到端方法跨域退化最小，但**仍必须做目标域微调**。

### 2.2 UMEG-Net：小样本 PES 的统一多实体图（AAAI 2026） ★Top3
- **链接**：https://arxiv.org/abs/2511.14186 。代码 https://github.com/LZYAndy/UMEG-Net
- **许可**：**无 LICENSE**，**无权重**。README 说明各数据集的姿态、球、场角关键点文件"即将上传"，目前缺失。
- **方法**：把人体骨架、球关键点、场地角点建成统一图，用空间 GCN 加无参数的多尺度时间移位（1、2、4 帧）编码。再以冻结的图模型作教师，用无标注 clip 把知识蒸馏到 VideoMAEv2 RGB 学生。
- **指标（100 个 clip，事件 F1 / Edit）**：

  | 数据集 | E2E-Spot 800MF | T-DEED 200MF | UMEG-Net | UMEG-Net 蒸馏版 |
  |---|---|---|---|---|
  | **ShuttleSet** | 42.7 / 54.6 | 33.8 / 41.7 | 49.2 / 64.0 | **59.1 / 69.0** |
  | F3Set-Tennis | 3.1 / 13.3 | 1.0 / 4.2 | 9.4 / 31.7 | 12.5 / 40.7 |

  这里的 F1 是包含击球类型的细粒度事件 F1，只看击球与否时会高得多。
- **接入 RallyIQ**：最贴合现有数据流（RTMO 的 17 点、TrackNet 球点、场角点），用**图模型替代人工设计的方向特征**，小样本设定也正好对应业余标注预算有限的情况。
  - 建议**按论文复现**（GCN 加 temporal shift 只需几百行），节点坐标先归一化到球场平面。
  - 双打时把 4 名球员都作为节点。
  - 蒸馏到 RGB 的阶段可以以后再做。
  - 同一方向还有 NUS 本科论文 *From Skeletons to Pixels*（arXiv 2604.22839）：骨架教师加退火伪标签，再蒸馏到 RGB 和光流。在 F3Set-Tennis 子集上用 50 个 clip 时 Edit 为 80.2，T-DEED 为 51.0。**没有代码**。

### 2.3 T-DEED（CVPRW 2024；SoccerNet BAS 2024 冠军）与 dude.k（BAS 2025 冠军）
- **T-DEED**：https://arxiv.org/abs/2404.05392 。代码 https://github.com/arturxe2/T-DEED ，checkpoint 在 Google Drive。
  - **许可**：**GPL-3.0**，产品中避免使用，实验可以。
  - **指标**：Tennis mAP@δ=0 为 56.0（200MF）、58.4（800MF），低于 E2E-Spot；δ=1 时约 97。
- **dude.k**：https://github.com/int8/dude.k 。冠军报告见 https://arxiv.org/abs/2508.19182
  - **许可**：无 LICENSE。
  - **方法与成绩**：在 T-DEED 基础上，把动作和队伍合并成单一预测头；先在 500 多场转播上预训练，再在比赛数据上微调；增强包括水平翻转并交换标签、随机裁剪、亮度抖动、**合成镜头平移**。Team-mAP@1 为 60.03，排名第一。
- **对 RallyIQ**：模型本身不建议直接用，但**"转播预训练 → 目标域微调 → 合成机位运动增强"**这一套是现成的迁移配方，可用于 2.1 和 2.2 的训练。

### 2.4 lRomul/ball-action-spotting（SoccerNet BAS 2023 冠军）
- **链接**：https://github.com/lRomul/ball-action-spotting
- **许可**：MIT，权重在 Google Drive。
- **方法与指标**：灰度帧三帧一组堆叠后送入 EfficientNetV2-B0 2D 编码器，再接 4 个 3D 倒残差块，即"慢融合"。配套技巧包括 0.6 秒正样本窗口、out-of-fold 难负例采样、模拟镜头运动和抖动的增强、EMA、focal loss，后处理为**Gaussian 平滑（σ=3）+ SciPy 峰值检测**。Challenge mAP@1 为 86.47。
- **接入 RallyIQ**：如果需要可商用的轻量 RGB 击球检测器，这是 MIT 许可下最完整的训练配方。其中的高斯平滑加峰值检测后处理，可以**直接套到现有打分器的逐帧输出上**，减少重复击球和漂移。工作量低。

### 2.5 F3Set / F3ED（ICLR 2025，网球快速、频繁、细粒度事件）
- **链接**：https://arxiv.org/abs/2504.08222 。代码 https://github.com/F3Set/F3Set
- **许可**：**无 LICENSE**。仓库 `f3set-model/` 目录内有训练好的模型。
- **内容**：网球数据，超过 1,000 种事件组合，帧级时间戳，示例为 25fps、720p；指标为事件 F1、元素 F1 和 Edit。另提供**标注工具**（annotation-tool）。
- **对 RallyIQ**：优先级中低。可借用它的**多层标注工具**为业余视频标击球加球种。其中的多标签分类思路（击球同时预测近端或远端、正手或反手等属性）适合后续扩展。

### 2.6 Multi-Modal Hit Detection and Positional Analysis in Padel（CVPRW 2024）
- **链接**：https://openaccess.thecvf.com/content/CVPR2024W/CVsports/papers/Decorte_Multi-Modal_Hit_Detection_and_Positional_Analysis_in_Padel_Competitions_CVPRW_2024_paper.pdf
- **许可**：**代码未发布**；数据集通过云盘链接公开，许可未声明。
- **方法与指标**：先用**音频 SED-net（CRNN）**检出击球时间窗，再在窗内跑 TrackNet 和 YOLO 姿态，按球与球员距离加权投票判断击球者。固定机位，共 2,377 个击球标注；事件 F1 为 **92%（σ 1.6）**，容差 250ms。
- **接入 RallyIQ**：手机和 DJI 素材都带音轨，音频起始点与机位几何**完全无关**，适合作为独立的候选源或确认器，用于召回。帧级定位仍由视觉完成，因为 250ms 在 30fps 下约为 ±4 帧。业余场馆往往多块场地同时打球，会有串音，需要靠与视觉候选的时间一致性来消歧。工作量中等。

### 2.7 Hsu et al., *Shot Refinement via Fusion of Shuttlecock Tracking and Hit Detection*（Sensors 2024）
- **链接**：https://pmc.ncbi.nlm.nih.gov/articles/PMC11244353 （doi:10.3390/s24134372）
- **许可**：**没有代码**，数据受限。
- **方法与指标**：轨迹拐点给出的击球（HD-T）与 YOLOv7 检出的挥拍动作（HD-A）用规则融合，例如每个挥拍段只保留一个轨迹击球、没有挥拍的轨迹击球直接丢弃。精度从 0.588 升到 **0.897**，F1 从 0.723 升到 **0.905**。数据为 BWF 和台湾 AI 竞赛的转播。
- **接入 RallyIQ**：用 RTMO 腕部速度峰值，或 RacketVision 拍头速度，做"挥拍门控"。规则可直接照搬，**1–2 天即可上线**，主要提高规则法（47/58）的精度。

### 2.8 kwyoke/Badminton-hit-detection（2022，业余数据集） ★Top5
- **链接**：https://github.com/kwyoke/Badminton-hit-detection
- **许可**：**无 LICENSE**。权重在 `mm_weights/` 和 Google Drive，数据也在 Google Drive。
- **内容**：
  - 三套标注数据：职业单打、**业余单打、业余双打**，全部为**底线单目机位**，标注包括球坐标、击球事件和球员框。
  - 方法：14 帧的球场、姿态和球坐标序列输入 GRU，输出三类：无击球、近端击球、远端击球。
  - 只用职业比赛训练，作者称能"一定程度"迁移到业余和双打。已知问题是广角机位下近端和远端容易混淆。
  - 指标为 mAP，只以图片形式给出，**未读取具体数值**。
- **接入 RallyIQ**：立即作为**第二个业余测试集**，补充现有的 58 击视频，也可作为小样本训练数据。由于没有许可，仅限内部评测；若用于训练商用模型，需先联系作者。"近端/远端"三分类的写法也可以直接借鉴。

### 2.9 ZSHYC/BadmintonTrackNet（MIT）与 TennisContactNet
- **BadmintonTrackNet**：https://github.com/ZSHYC/BadmintonTrackNet
  - **许可**：MIT。没有自带权重，使用官方 TrackNetV3 权重。
  - **内容**：在 TrackNetV3 上加入运动学规则（速度反向、加速度峰值等）做**高召回**的击球和落地候选，再用 **BounceNet** 二阶段分类器过滤。BounceNet 支持纯轨迹、纯视觉或融合输入。仓库还带半自动标注工具和 Dash 误差分析界面。**未报告事件指标**。
  - **对 RallyIQ**：架构与现有方案几乎一样。可以直接借用 MIT 代码中 BounceNet 的**视觉分支**：在候选帧附近裁出图块送入 CNN，补上纯轨迹特征跨机位失效的问题。
- **TennisContactNet**：https://github.com/ZSHYC/tennis-contact-rgb
  - **许可**：**无 LICENSE**，权重 `weights/best.pt` 在仓库内。
  - **内容**：纯 RGB 的网球击球和落地检测，结构为 X3D-S 加逐帧通道注意力和 BiGRU，约 6.8M 参数，事件匹配容差 ±5 帧。
  - **对 RallyIQ**：仅作为小型 RGB 击球检测器的结构参考。

### 2.10 其他（一句话）
- **Santra et al. 2025（Sony）**，https://arxiv.org/abs/2503.00147 ：ASTRM 模块、Bi-GRU、SoftIC 损失。按作者复现，Tennis mAP@δ=0 为 61.0，E2E-Spot 为 45.3。**无代码**，可借鉴处是用 SoftIC 处理类别不平衡。
- **BST**（CVPRW 2026），https://github.com/Va6lue/BST-Badminton-Stroke-type-Transformer ：MIT，权重在 Google Drive。用 RTMPose、TrackNetV3 和球场位置做**球种分类**，ShuttleSet 25 类准确率约 0.83。属于击球检测之后的下游模块，不解决当前瓶颈。
- **FineBadminton**（ACM MM 2025），https://arxiv.org/abs/2508.07554 ：BWF 转播数据，标注流水线中用微调过的 VideoMAE 做击球检测。**没有代码链接**，相关度低。
- **tt_sounds**（图宾根），https://github.com/cogsys-tuebingen/tt_sounds ：乒乓球击球声检测。能量峰值检测的 P 为 0.98、R 为 0.95–1.0，接触面分类 F1 为 0.97。无 LICENSE。可以说明短促击球声的起始点检测本身很容易，难点在于场馆串音。
- **UGLF**（IJCNN 2024，BSD-3，无权重）：针对足球 SoccerNet-v2，用 GLIP 提取局部实体特征。**不推荐**。
- **P2ANet**（2022，乒乓球转播动作检测）：时间早、只有转播数据，**不推荐**。

---

## 3. 广播 → 业余机位：域迁移和小样本技巧

**证据**：*Beyond the Premier*（CVPRW 2024，https://openaccess.thecvf.com/content/CVPR2024W/CVsports/papers/Cabado_Beyond_the_Premier_Assessing_Action_Spotting_Transfer_Capability_Across_Diverse_CVPRW_2024_paper.pdf ）。SoccerNet 训练的模型零样本迁移到 AI 自动机位的业余比赛时，E2E-Spot 的紧 mAP 从 61.8 跌到 **19.6**，ViSwin 也只有 25.9。基于特征的方法跌得更多。这与 RallyIQ 学习打分器只找回 34/58 的现象一致：**不做目标域适配就不行；端到端方法退化最小**。论文没有代码。

按投入产出比排序的技巧：
1. **把特征换到与机位无关的坐标系**（参考 Where Is The Ball、FreeTrack 的运动平面、MonoTrack）。用场角单应把球和姿态坐标映射到球场平面，或改用射线规范表示，再计算速度和方向变化等特征。这样 ShuttleSet 上训练的打分器在低机位下的分布偏移会小很多。前提是做场地标定，例如 FreeTrack 的 6 点法。**工作量低到中，预计对学习打分器收益最大。**
2. **合成多机位训练数据**（参考 Where Is The Ball 的 PhysX、Uplifting TT 的 MuJoCo）。用带阻力的羽毛球动力学，在击球点之间生成 3D 轨迹，按随机的业余机位（低三脚架、底线后、侧面；30–60fps；加噪声和漏检）投影到 2D，用来训练或扩充击球打分器。也可以用 MonoTrack 式物理拟合把 ShuttleSet 回合重建成 3D，再从业余视角重投影，这样标签是现成的。工作量中等。
3. **先转播预训练，再小样本微调，并加机位运动增强**（参考 dude.k、lRomul）：模拟镜头平移和抖动，做水平翻转并交换近端、远端标签。
4. **关键点图加小样本**（UMEG-Net）：100 个 clip 级别的标注就能明显超过 RGB 端到端方法。
5. **中值背景输入加多运动联合训练**（RacketVision、TrackNetV3/V5 的 `bg_mode`）：固定机位的业余素材最适合用中值背景。
6. **遮挡增强加可见性加权损失**（TOTNet）：针对侧机位下手臂和身体的遮挡。
7. **模糊中心标注**（BlurBall）：业余 30fps 拖影严重，统一标在拖影中点。
8. **半自动标注**（ETH 的 GMM 背景减除加人物去除、kwyoke 的流水线、F3Set 的标注工具），再配合**双检测器一致性伪标签**（TrackNetV3 与 WASB 同时高置信才采纳，这是本报告的建议，不是来自某篇论文），在自有素材上做自训练，降低人工标注量。
9. **独立模态兜底**：音频起始点（Padel CVPRW 2024）和挥拍门控（Hsu 2024）都与机位几何无关，可以和视觉打分做晚融合。

---

## 4. 许可证速查

| 项目 | 代码许可 | 权重 | 可否直接商用 |
|---|---|---|---|
| RacketVision | MIT | HF（模型卡未核实），数据 MIT | 基本可以，需复核 HF 模型卡 |
| WASB | MIT | Google Drive，随仓库 | 可以，训练数据条款另计 |
| BlurBall | MIT | Nextcloud | 代码可以，数据许可未声明 |
| TOTNet | MIT | 无 | 代码可以，TTA 数据仅限研究 |
| TrackNetV5（NYCU / gino6178） | 无 | 有（releases/） | **否**，需授权 |
| TrackNetV5（武汉 SDK） | 专有 | 不公开 | **否** |
| ETH shuttle_detection | AGPL-3.0 | 未提及 | **否**（强 copyleft） |
| Uplifting Table Tennis | GPL-3.0 | 见仓库 | **否**（copyleft） |
| FreeTrack | 无 | weights/ 目录 | **否** |
| Where Is The Ball | 未发布 | 未发布 | — |
| AdaSpot | MIT | HF（未核实） | 基本可以 |
| E2E-Spot | BSD-3 | 权重仓库无许可 | 代码可以，权重需自训 |
| T-DEED | GPL-3.0 | Google Drive | **否**（copyleft） |
| dude.k | 无 | — | **否** |
| lRomul BAS 2023 | MIT | Google Drive | 可以 |
| F3Set / F3ED | 无 | 仓库内 | **否** |
| UMEG-Net | 无 | 无 | **否**，可按论文复现 |
| kwyoke hit-detection | 无 | Google Drive | **否**，仅内部评测 |
| BadmintonTrackNet（ZSHYC） | MIT | 使用 TrackNetV3 权重 | 可以 |
| TennisContactNet | 无 | 仓库内 | **否** |
| BST | MIT | Google Drive | 可以 |
| Padel CVPRW 2024 / Hsu Sensors 2024 / Santra 2025 | 未发布 | — | 仅能借鉴思路 |

附带核实：RallyIQ 当前使用的 **qaz812345/TrackNetV3** 的 LICENSE 是 MIT，并**明确写明覆盖 "pretrained model checkpoints"**。GitHub 页面显示 NOASSERTION，是因为文本被改动过、未被自动识别，不影响使用。

---

## 5. 局限和未核实项
- 本次 WebFetch 配额用完，以下内容**未读到**：*Player Disjoint Multi-Task Tennis Stroke Analysis from Monocular Pose Sequences*（arXiv 2606.15992，可能涉及纯姿态的击球时刻检测，值得补查）；1000 Rallies 的代码情况；AdaSpot 完整的分数据集表格（只读到摘要中的增益数字）。
- HuggingFace 在本环境无法访问，RacketVision 和 AdaSpot 的 HF 模型卡许可未核实。
- kwyoke 数据集的具体 mAP 只有图片形式，没有读出数值；其中业余视频的分辨率和帧率需要下载后确认。
- TrackNetV5（NYCU）仓库 README 自称是"reproduction"，没有确认是否为作者官方仓库，不过作者项目页 gino6178.github.io 与仓库归属一致。
