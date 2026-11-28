# RallyIQ 调研：姿态 / 3D / 多目标跟踪 / 球拍 / 步法（2023–2026 新工作）

> 注：本文件是原始调研稿。许可证和数字已另行逐条核查，更正见上级目录 `2026-11-28-web-research.md` 第三节和 `verification.json`；两者不一致时以核查结果为准。


调研日期：2026-11-28。已排除任务中列为“已知”的工作（RTMO/RTMPose、YOLOv8-pose、ByteTrack、OSNet ReID 思路、MediaPipe、IsaacChen89 split-step、Va6lue Key-Frame-Detection、TemPose、BST）。

**许可证核验方式**：代码许可证都用 GitHub API 的 `license.spdx_id` 字段查过；标为 NOASSERTION 或 null 的仓库，再读原始 LICENSE 或 README 原文。权重许可证以 Hugging Face 模型卡或 README 的原文为准。没能核验的项会标“未核验”。
**局限**：调研后半段 WebFetch 的配额用完了（多个 agent 共用一个预算），因此部分数字取自 GitHub README 或论文 PDF 的原文。CameraHMR 的权重条款、CalTennis 与 BFMD 的数据许可证这几项没能核验，文中已标出。

---

## 0. TL;DR：最值得落地的 5 件事

| # | 做什么 | 对应痛点 | 许可证 | 工作量 |
|---|---|---|---|---|
| 1 | **球场 ROI 过滤 + GTA-Link 离线 tracklet 拆分/合并**。在现有 tracker 输出上做后处理，再加“近端/远端阵营”和“最多 2 或 4 个 ID”这两条约束 | 双打 ID 切换、邻场路人 | GTA-Link 代码为 MIT | 小：纯后处理，不需要重新训练 |
| 2 | **RacketVision 球拍 5 点姿态**（RTMDet+RTMPose，含羽毛球数据和权重）。击球帧的击球者判定从“最近手腕”改为“最近拍头（top 点）”，手腕被遮挡时作为备选 | 击球时刻 hitter 被遮挡 | 代码 MIT，HF 数据集卡标 MIT | 中：需要在业余画面上补少量标注再微调 |
| 3 | **远端半场 ROI 二次 top-down 推理：RTMW-x 384×288（133 点全身，含 6 个足部点）** | 远端球员小、被网遮挡；COCO-17 没有脚尖和脚跟点，限制了 split-step 和触地判断 | mmpose 代码为 Apache-2.0（权重随 mmpose 发布，训练数据条款商用前需复核） | 小到中 |
| 4 | **3D 技术分析改用许可证友好的 GEM-X（NVIDIA，SOMA 人体模型）或 SAM 3D Body（MHR 人体模型）**，不用 SMPL 系方法 | 技术动作 3D 角度、触地与稳定性 | GEM-X：代码 Apache-2.0，权重为 NVIDIA Open Model License（模型卡写明可商用），SOMA 为 Apache-2.0。SAM 3D Body：SAM License（允许商用，有贸易管制条款），MHR 为 Apache-2.0 | 中到大（离线、需要 GPU） |
| 5 | **CAMELTrack：可学习的关联模块，能吃 pose 关键点线索**。RallyIQ 本身已有姿态，接入成本低 | 同色衣服的双打 ID 切换 | 代码 Apache-2.0（HF 权重许可证未标） | 中：要在羽毛球数据上训练，论文称单张消费级 GPU 约 1 小时 |

次优先：SAM 3.1 掩码跟踪（遮挡和交叉时作为离线身份兜底）；CalTennis 的 footwork/stability 指标（作为 3D 步法 QA）；精英 split-step 时序先验；用 MultiSenseBadminton 的足底压力数据给触地检测做真值。

---

## 1. 2D 姿态（小目标、遮挡、许可证友好）

### 1.1 RTMW / RTMW3D（OpenMMLab，2024）
- 论文：https://arxiv.org/abs/2407.08634 ；代码：https://github.com/open-mmlab/mmpose （projects/rtmpose）
- 许可证：mmpose 代码 **Apache-2.0**（已核验）。权重随 mmpose 发布，没有单独的权重许可证；训练用了 14 个公开数据集，其中一部分是研究用途，**商用前需复核训练数据条款**。
- 精度（COCO-WholeBody v1.0，AP）：RTMW-x 384×288 全身 70.2 / 身体 76.3 / **足部 79.6**；RTMW-l 384×288 全身 70.1。RTMW3D-x 的 COCO-WholeBody 为 68.0，H3WB MPJPE 为 0.057（论文未写单位）。
- 速度：RTMW-l 384×288 在 CPU（ONNXRuntime）上 47.6 ms 一个人；RTMPose-l 同输入为 44.6 ms。
- **接入 RallyIQ**：保留 RTMO 做全图检测和粗姿态，对**远端半场 ROI**（或所有低于某个像素高度的人框）放大后用 RTMW-x 做二次推理。133 点里有**脚跟、大脚趾、小脚趾**，split-step 的离地/落地、触地点和步幅都能直接用，比 COCO-17 只有脚踝点精确得多。RTMW3D 可以低成本地给出每个关节的相对深度，适合作为 3D 的“轻量档”（注意它的 3D 训练数据 H3WB 来源于 H36M，H36M 是研究用途许可）。

### 1.2 ViTPose++（Hugging Face transformers 版）
- 论文：https://arxiv.org/abs/2212.04246 ；代码：https://github.com/ViTAE-Transformer/ViTPose ；HF：https://huggingface.co/usyd-community/vitpose-plus-small
- 许可证：GitHub 代码 **Apache-2.0**；HF 上 usyd-community 的权重卡标 **Apache-2.0**（均已核验）。
- 精度：HF 卡写明 ViTPose++ 在 COCO test-dev 上 81.1 AP；ViTPose-H 为 80.9。
- 速度：B/S 档可以实时，H 档较慢（卡片里只有图，没有文字数值）。
- **接入**：`transformers` 一行就能加载，适合作为远端小目标 top-down 精修的另一个选择，也可以作为 RTMW 的交叉验证基线。下文的双打论文、OpenCap Monocular、GEM-X 的 EMDB 评测都用 ViTPose 作 2D 输入，说明它和 3D 下游方法搭配最成熟。

### 1.3 DETRPose（2025，Apache-2.0）
- 论文：https://arxiv.org/abs/2506.13027 ；代码：https://github.com/SebastianJanampa/DETRPose （**Apache-2.0**，已核验；README 带 HF 权重）
- 精度：COCO val2017 上 DETRPose-X 73.3 AP、-L 72.5 AP。**CrowdPose test** 上 -X 75.1 AP（AP_hard 68.1），-L 73.3 AP（AP_hard 66.1）。
- 速度（README 所列延迟）：-L 9.5 ms，-X 13.3 ms，-N 2.8 ms。
- **接入**：单阶段、不需要 NMS，在拥挤和遮挡指标（CrowdPose AP_hard）上表现不错。COCO 精度和 RTMO-l（据 RTMO 论文约 74.8 AP）差不多，**不建议为了精度替换 RTMO**。如果将来需要一个 Apache-2.0 的端到端检测加姿态模型、并想在羽毛球数据上微调（比如“球员 + 球拍”多类别），它是很干净的底座。

### 1.4 RF-DETR Keypoint（Roboflow，2026 预览版）
- 文档：https://rfdetr.roboflow.com/latest/learn/run/keypoints/ ；代码：https://github.com/roboflow/rf-detr （**Apache-2.0**，已核验；文档写明 Keypoint 模型为 Apache 2.0）
- 精度和速度：COCO AP50:95 为 71.8；T4（TensorRT FP16，batch 1）上 **9.7 ms**；40.7M 参数，输入 576×576。
- 注意：目前是 **Preview**，API 和权重可能变动。文档同时说明 YOLOv8、YOLO11、YOLO26 都是 **AGPL-3.0**。
- **接入**：可以自定义关键点布局再微调，比如训练一个同时输出“人 17 点 + 球拍 5 点”的模型。它和 DETRPose 都可以作为 AGPL 的 YOLO-pose 的替代。

### 1.5 Sapiens2（Meta，ICLR 2026）与 Sapiens v1：许可证要特别注意
- 论文：https://arxiv.org/abs/2604.21681 ；代码：https://github.com/facebookresearch/sapiens2 ；v1：https://github.com/facebookresearch/sapiens
- 许可证（已读原文）：
  - **Sapiens v1：CC-BY-NC-4.0，不可商用。**
  - **Sapiens2：自定义的 “Sapiens2 License”**。授予免版税的使用、修改和分发权，没有写“非商用”，但**禁止“biometric processing（生物特征处理）”、监控、对 Sapiens 材料中的个人做识别或再识别、未经同意推断健康信息**，Meta 还保留审计权。RallyIQ 做运动员姿态分析和 ReID，**有落入“biometric processing”的风险，需要法务确认**。
- 精度和规格：模型规模 0.1B–5B（pose 只提供 0.4B/0.8B/1B/5B）；**308 个全身关键点**；原生分辨率 1024×768；论文称 pose 比 v1 高 4 mAP。已集成进 transformers（`AutoModelForPoseEstimation`）。
- 速度：0.4B 档约 1.26 TFLOPs 一帧一人，**不适合全量视频推理**。
- **接入**：只建议作为**离线标注器或教师模型**，比如给远端小目标生成伪标签，再蒸馏到 RTMW/RTMO。前提是法务认可它的许可证。

> AGPL 提醒：YOLO11-pose / YOLO26-pose（Ultralytics）都是 AGPL-3.0。下文 Fast SAM 3D Body 默认用 `yolo11m-pose` 做检测器，商用时要换成 RTMO、DETRPose 或 RF-DETR。

---

## 2. 单目 3D 人体姿态与网格（技术分析）

### 2.1 GEM-X（NVIDIA，2025-05 首发，2026-10 更新）：首选的商用友好方案
- 代码：https://github.com/NVlabs/gem-x （**Apache-2.0**，已核验）；权重：https://huggingface.co/nvidia/GEM-X （**NVIDIA Open Model License**，模型卡原文为 “ready for commercial or non-commercial use”）；人体模型 SOMA：https://github.com/NVlabs/SOMA-X （**Apache-2.0**，已核验）
- 能力：单图 Body Pose、Hand Pose，以及**视频 Body Motion**（输出相机系和世界系运动，带 Contact/IK 后处理）；77 个关节；支持 ONNX/TensorRT；模型卡把“体育中的生物力学分析”列为预期用途。
- 精度：GEM-pose 在 3DPW 上 MPJPE 62.2、PA-MPJPE 39.6、PVE 72.9，EMDB-1 MPJPE 67.8。GEM-motion 在 EMDB-2 上 W-MPJPE100 267.2、WA-MPJPE100 112.8、RTE 2.45%、**足部滑动 4.22 mm**（开启 Contact/IK）。
- 速度：未给 FPS（标称 A100 推理）。
- **接入**：用 RallyIQ 现有的 tracker 结果（人框和 ID）离线逐球员跑 motion-video，得到世界坐标下的 3D 关节，再用球场 homography 对齐到地面坐标。可以分析挥拍链（肩、肘、腕角度）、重心高度、触地相位。这是**唯一同时满足视频时序、世界坐标、触地约束和可商用**的方案。注意它的 NVIDIA 姊妹项目 GENMO 是**非商用**许可，不要混用。

### 2.2 SAM 3D Body（Meta，2025-11，CVPR 2026）+ MHR
- 论文：https://arxiv.org/abs/2602.15989 ；代码：https://github.com/facebookresearch/sam-3d-body ；MHR：https://github.com/facebookresearch/MHR
- 许可证：代码和权重为 **SAM License**（已读原文：免版税授权，含专利授权，允许销售；限制只涉及贸易管制和军事等用途）；**MHR 为 Apache-2.0**（已核验）。HF 权重需要申请访问。
- 精度（DINOv3-H+，840M）：3DPW MPJPE 54.8、EMDB MPJPE 61.7、RICH PVE 60.3、COCO PCK@.05 86.5。
- 速度：Fast SAM 3D Body 论文称原版**每张图要数秒**。
- 关键特性：**可以用 2D 关键点和掩码作为提示**，所以 RTMO/RTMW 的关键点可以直接喂进去，约束被遮挡的肢体。
- 第三方验证（CVBW@CVPR 2026 论文 14，智能手机视频对比 OpenCap 动捕）：步行下肢关节角 MAE 为 SAM3D 5.98°、CameraHMR 6.09°；坐站转换为 7.95° 对 5.92°；膝屈误差最小，额状面和水平面误差最大。
- **接入**：在关键帧（击球帧、split-step 落地帧）上做高质量单帧 3D，并用 RallyIQ 的 2D 关键点作提示，适合生成“技术快照”。

### 2.3 Fast SAM 3D Body（ECCV 2026）
- 论文：arXiv 2603.15603 ；代码：https://github.com/yangtiming/Fast-SAM-3D-Body （**MIT**，已核验；底层权重仍是 SAM License）
- 速度：端到端最高加速 **10.9×**；在 RTX 5090 上约 **65 ms/帧**（遥操作演示）；MHR→SMPL 的映射改成前馈网络，加速超过 10,000×。
- 注意：默认检测器是 **yolo11m-pose（AGPL）**，要替换。转成 SMPL 输出会重新引入 SMPL 的非商用许可，**应直接使用 MHR 输出**。
- **接入**：如果要对整段回合逐帧跑 3D，这是把 SAM 3D Body 降到可用延迟的现成方案。

### 2.4 SAM-Body4D（2025-12，免训练视频版）
- 论文：https://arxiv.org/abs/2512.08406 ；代码：https://github.com/gaomingqi/sam-body4d （**MIT**，已核验；依赖 SAM 3 和 SAM 3D Body，均为 SAM License，还依赖 Diffusion-VAS，其许可证未核验）
- 方法：SAM 3 生成跨帧身份一致的掩码；**Diffusion-VAS 做非可见（amodal）补全**，即 IoU<0.7 时判定为遮挡并补全；SAM 3D Body 以掩码为提示逐帧重建，再做 Kalman 平滑，并把体型锁定在首个可见帧。
- 精度：论文**只有定性结果**。速度：A100 上 90 帧、5 人的视频比逐帧串行快约 2 倍（不含补全器）。
- **接入**：思路直接对应“**被网遮挡的远端球员**”和“**被搭档遮挡的击球者**”。可以先只借用“掩码跟踪 + 遮挡检测 + 体型锁定”这套流程，补全器按需开启。

### 2.5 SMPL 系与其他视频 3D 方法：精度强，但许可证大多不能商用
| 方法 | 链接 | 代码许可证（已核验） | 权重 / 依赖 | 要点 |
|---|---|---|---|---|
| WHAM (CVPR'24) | github.com/yohanshin/WHAM | MIT | **需要 SMPL（MPI 注册，非商用）** | CalTennis 中足速一致性最好（0.72 m/s）；OpenCap Monocular 的底座 |
| TRAM (ECCV'24) | github.com/yufu-wang/tram | MIT | SMPL + DROID-SLAM（第三方许可证） | 适合移动相机；RallyIQ 是固定机位，收益有限 |
| GVHMR (SIGGRAPH Asia'24) | github.com/zju3dv/GVHMR | **仅限教育、研究、非营利** | SMPL | 不可商用 |
| PromptHMR (CVPR'25) | github.com/yufu-wang/PromptHMR | **非商用科研** | SMPL-X | CalTennis 中姿态和平移一致性最好（PA-MPJPE 0.084 m） |
| GENMO (NVIDIA) | github.com/NVlabs/GENMO | **NVIDIA OneWay Noncommercial** | — | CalTennis 中足高和稳定性最一致；商用请换 GEM-X |
| CameraHMR | github.com/pixelite1201/CameraHMR | **仓库未声明许可证** | SMPL | 生物力学精度与 SAM3D 相近（见 2.2） |
| SMPLer-X | github.com/caizhongang/SMPLer-X | **S-Lab License 1.0（非商用）** | SMPL-X | — |
| NLF (NeurIPS'24) | github.com/isarandi/nlf | MIT | **README 称模型仅供非商用研究** | — |
| Multi-HMR | github.com/naver/multi-hmr | **NAVER 非商用** | SMPL-X | — |
| 4D-Humans/PHALP | github.com/shubham-goel/4D-Humans | MIT | SMPL | — |
| MotionBERT | github.com/Walter0807/MotionBERT | Apache-2.0 | 权重用 H36M/AMASS 训练（数据仅限研究） | 见 2.6 的“MPJPE 悖论” |
| OpenCap Monocular | github.com/utahmobl/opencap-monocular | **PolyForm Noncommercial** | WHAM + SMPL | WHAM 加物理约束优化，再接 OpenSim 逆运动学；作者指出**腾空阶段（跳跃）触地概率会退化** |

**结论**：SMPL 系方法只适合做研究对照或内部基准。产品化路径是 GEM-X（SOMA）或 SAM 3D Body（MHR）。

### 2.6 生物力学有效性证据：决定 3D 怎么评估
- **AthletePose3D（CVPRW 2025）**：https://arxiv.org/abs/2503.07499 ，代码和数据见 https://github.com/calvinyeungck/AthletePose3D （**仅限非商用科研**，需签协议）。约 1.3M 帧，项目为田径和花滑，**没有持拍类运动**。在 H36M 上训练的模型在运动员动作上 MPJPE 约 214 mm，用该数据微调后降到约 65 mm（降幅超过 69%）。关节角波形与动捕的相关系数：上肢 r=0.89、下肢 r=0.81；**角速度相关性很弱**（0.29 和 0.12）。
- **“Does Lower MPJPE Mean Better Biomechanics?”（CVBW 2026 论文 12）**：https://cvbw2026.github.io/assets/papers/12.pdf 。在 AthletePose3D 上，MotionBERT 的 MPJPE 最好（37.2 mm），关节角误差却最差（25.19°）；SimpleBaseline3D 的 MPJPE 最差（58.5 mm），角度误差最好（19.21°）。
- **对 RallyIQ 的启示**：评估 3D 技术分析要**用关节角和时相指标**，不要只看 MPJPE；角速度类指标（如挥拍速度）在单目条件下暂不可靠，展示时应标为“估计”。

---

## 3. 多目标跟踪（外观相似、遮挡、路人）

### 3.1 GTA-Link：全局 tracklet 关联（ACCV-W 2024）
- 论文：https://arxiv.org/abs/2411.08216 ；代码：https://github.com/sjc042/gta-link （**MIT**，已核验）
- 方法：**离线后处理，与 tracker 无关**。Splitter 用改进的 DBSCAN 对 tracklet 内的 ReID 特征聚类，把混入多个身份的 tracklet 拆开；Connector 用余弦距离加出入点空间约束做层次聚类合并。ReID 用 OSNet（SportsMOT 训练）。
- 精度：接在 Deep-EIoU 之后，SportsMOT 上 HOTA 77.21→**81.04**、IDF1 79.81→**86.51**；SoccerNet 上 HOTA 79.41→83.11。
- **接入（Top-1）**：RallyIQ 本来就是离线分析，可以直接接在现有 tracker 之后。再加两条羽毛球先验，效果会更好：① 同一 ID 不跨网（近端/远端阵营由 homography 判定，双打两人同侧）；② 每侧最多 1 或 2 个 ID，合并时把簇数上限设为 2 或 4。

### 3.2 CAMELTrack（2025）
- 论文：https://arxiv.org/abs/2505.01257 ；代码：https://github.com/TrackingLaboratory/CAMELTrack （**Apache-2.0**，已核验；HF 权重许可证未标）
- 方法：数据驱动的多线索关联（外观、框、**关键点**），基于 tracklab 框架。
- 精度：SportsMOT HOTA **80.4**（Deep-EIoU 77.2，MOTIP 71.9）；DanceTrack 上加关键点后 HOTA 66.1→**69.3**（+3.2），SportsMOT 上关键点基本没有增益。
- 速度：CAMEL 关联本身 18 ms/帧；整条流水线（YOLOX 24 ms + RTMPose 17 ms + BPBReID 16 ms + CAMEL 18 ms）在 MOT17 上约 13 FPS。
- **接入**：RallyIQ 已经有 RTMO 关键点，正好对应 DanceTrack 那种“衣服相同、动作复杂”的场景，也就是关键点最有用的场景。需要用少量羽毛球双打数据训练关联器。

### 3.3 其他 sports MOT 对比（作为参考）
| 方法 | 链接 | 代码许可证 | SportsMOT HOTA | 备注 |
|---|---|---|---|---|
| Deep-EIoU (WACV-W'24) | github.com/hsiangwei0903/Deep-EIoU | **未声明（默认保留所有权利）** | 77.2 | YOLOX + OSNet；GTA-Link 的基线 |
| MOTIP (CVPR'25) | github.com/MCG-NJU/MOTIP | Apache-2.0 | 71.9（据 CAMELTrack 表格） | 端到端 ID 预测，对小数据场景不友好 |
| Hybrid-SORT (AAAI'24) | github.com/ymzis69/HybridSORT | MIT | — | DanceTrack 65.7；“弱线索”（置信度、高度状态）思路可以借鉴 |
| OA-SORT (arXiv 2603.06034, 2026) | 未见代码 | — | 75.2 | 免训练，基于框底边做遮挡深度排序；24.6 ms/帧 |
| SelfTrEncMOT (BMVC'25) | github.com/Milad-Khanchi/SelfTrEncMOT | 未声明 | 76.4 | Depth Pro + SAM2 掩码；深度估计约 0.3 s/帧 |

### 3.4 SAM 3 / SAM 3.1 Object Multiplex（Meta，2025-11 / 2026-03）
- 论文：https://arxiv.org/abs/2511.16719 ；代码：https://github.com/facebookresearch/sam3 （**SAM License**，允许商用，已读原文）
- 能力：文本、框、点提示的视频分割和跟踪，848M 参数；presence token 专门改善 “a player in white” 和 “a player in red” 这类近似描述的区分。SAM 3.1 的 Object Multiplex 在 128 个目标时比 2025-11 版快约 7 倍（H100）。
- **接入**：作为**离线身份兜底**。在双打交叉、遮挡的片段（tracker 置信度下降时）用球员框提示 SAM 3.1 做掩码传播，用像素级连续性修正 ID；它也是 SAM-Body4D 的第一步。代价是 GPU 开销大，只在疑难片段触发。

### 3.5 双打羽毛球：“Bridging the Gap: Doubles Badminton Analysis with Singles-Trained Models”（2025）
- 论文：https://arxiv.org/abs/2508.13507 ；代码：https://github.com/100-heon/badminton_double_analysis （**仓库未声明许可证**；用的是 YOLOv11x，属 AGPL）
- 方法：YOLOv11x 检测人，**另一个模型检测球场四角，生成球场 ROI 过滤裁判和观众**；BoT-SORT 加速度外推，丢失时间不超过 10 帧且重现位置在 200 px 内就恢复原 ID；ViTPose 17 点；ST-GCN 对比学习编码器，近端和远端分开建模，用来判定谁在击球。
- 精度：只在两场双打上验证，击球者判定准确率 0.6686（YOLO 基线 0.5941），样本很小。
- **接入**：“球场 ROI 过滤 + 短时速度外推找回 ID”可以直接照搬到 RallyIQ，用于解决**邻场路人**问题。具体做法是：脚点经 homography 落在球场多边形加一定边距内才保留，并在 tracklet 级别统计在场比例。

### 3.6 “运动学指纹”：姿态可以区分人（CVBW 2026 论文 11）
- 论文：https://cvbw2026.github.io/assets/papers/11.pdf ；数据和代码：https://github.com/jasnwag/tennis_serve_dataset （**CC BY 4.0**，已读原文）
- 内容：RTMDet + RTMPose + MotionBERT，处理 2024 美网 5,966 次发球、109 名球员；**仅用关节角轨迹，随机森林识别球员的准确率就达到 99.2%**。
- **接入**：给 ReID 加一路**不依赖衣服的特征**：骨长比例（相当于身高和臂展）、持拍手、站姿宽度等。双打同色球衣时用它打破平局，在 GTA-Link 的 connector 或 CAMELTrack 里作为附加线索。注意这是闭集分类，结论不能直接外推到开放集 ReID。

---

## 4. 球拍检测与球拍姿态

### 4.1 RacketVision（AAAI 2026 Oral）
- 论文：https://arxiv.org/abs/2511.17045 ；代码：https://github.com/OrcustD/RacketVision （**MIT**，已核验）；数据：https://huggingface.co/datasets/linfeng302/RacketVision （数据卡标 **MIT**）；模型：huggingface.co/linfeng302/RacketVision-Models（未单独核验）
- 规模：羽毛球 461 个片段、114,753 帧，加上网球和乒乓球，总计 435K 帧。每支球拍标注框和 **5 个关键点**（top、bottom、handle、left、right）。坐标按 1920×1080 记录，源视频看起来是转播视角（未确认）。
- 精度（多运动 RTMDet+RTMPose 模型，羽毛球）：PCK@0.2 **88.5%**、MPJPE 5.00 px、mOKS 0.668；分点为 top 99.4、bottom 99.7、handle 97.3、left 74.6、right 75.5。羽毛球球体跟踪（MS-TrackNetV3）P 0.915、R 0.865。轨迹预测用交叉注意力融合球拍信息后，长轨迹 ADE 有改善，短轨迹改善很小。
- **接入（Top-2）**：
  1. **击球者判定**：击球帧附近 ±k 帧内，以“拍头 top 点到羽毛球的距离”为主、“手腕距离”为辅。手腕被躯干或搭档遮挡时，拍头往往仍然可见。
  2. **准备动作特征**：举拍高度和拍面朝向（top 到 handle 的向量）可以作为 split-step 和击球准备的辅助信号。
  3. 域差：业余手机或 DJI 画面需要补标少量帧（建议 300–1000 帧）再微调。RacketVision 的完整训练和评测流水线可以直接复用。

---

## 5. 步法与移动分析（论文、数据集）

### 5.1 CalTennis（2026-06）：步法和稳定性指标
- 论文：https://arxiv.org/abs/2606.20542 ；数据：huggingface.co/datasets/demalenk/caltennis （**数据许可证未核验**；论文本身为 CC BY 4.0）
- 规模：约 51 小时、超过 11M 帧、40 名球员；**2–6 部 iPhone 架在 1.65 m 三脚架上**，60 Hz 同步。机位和 RallyIQ 的业余场景很接近。
- 评测了 TRAM、GVHMR、GENMO、WHAM、PromptHMR。各方法根关节平移的跨视角分歧约 0.9–3.6 m，姿态分歧约 11 cm；**所有方法在深度和触地上都不稳定**。
- 新指标：**footwork**（足部速度和足高的跨视角一致性，即“滑步”“漂浮”）；**stability**（重心投影到触地足凸包的距离）。
- **接入**：把 footwork 和 stability 两个指标作为 RallyIQ 3D 步法模块的 QA 和展示指标；如果数据许可证允许，用 CalTennis 做业余机位的回归测试集。

### 5.2 世界顶级羽毛球运动员的 split-step 特征（PMC，2024–25）
- 链接：https://pmc.ncbi.nlm.nih.gov/articles/PMC11698525/
- 方法：BWF 转播固定机位，30 fps 下人工标注双足拇指根部，再用 homography 映射到场地坐标（误差 44.8±22.6 mm）。
- 发现：从对手击球到一只脚离地的反应时间约 **0.24–0.25 s（30 fps 下约 7–8 帧）**；站姿宽度约为**身高的 0.5 倍**；split-step 位置靠近中心位，但离散度大（SD 24–34 cm）。
- 局限：只有 3 名男单精英、只看正手后场方向；统计报告前后不一致（同一个 F 值对应了不同 p 值）。
- **接入**：作为 split-step 检测器的先验或合理性检查，例如“对手击球后 0.1–0.4 s 内出现离地事件”“站宽和身高之比 0.3–0.7”。业余球员的数值会更慢、更窄，**应作为可视化参照，不要当硬阈值**。

### 5.3 羽毛球数据集速览
| 数据集 | 链接 | 规模和内容 | 与 RallyIQ 的关系 | 许可证 |
|---|---|---|---|---|
| **VideoBadminton** (2024) | arxiv.org/abs/2403.12385 | 7,822 段，18 类击球，60 fps、1280×960，相机在底线后约 2 m、高约 4.5 m | 机位最接近（后场视角）；**没有步法类别** | 论文为 CC BY-NC-SA；数据称“供研究用” |
| **FineBadminton** (2025) | arxiv.org/abs/2508.07554 ，finebadminton.github.io/FineBadminton | 3,215 个回合、33,325 拍，BWF 单打转播 25 fps；有 “Moving Recognition” 任务 | 战术和击球语义；只有单打 | 未声明 |
| **BFMD** (2026) | arxiv.org/abs/2603.25533 | 19 场完整转播（**7 场双打**），16,751 个击球事件，球员框和姿态为自动生成 | 可以用来测**双打击球事件**召回 | 未声明（链接未给出） |
| **MultiSenseBadminton** (Sci Data 2024) | nature.com/articles/s41597-024-03144-z ，github.com/dailyminiii/MultiSenseBadminton | 25 人、7,763 次挥拍；IMU、EMG、**足底压力**、3 路外部相机 | 足底压力可以作为**触地和 split-step 时序的真值**；没有步法标签 | 代码 MIT；数据许可证未核验 |
| Mendeley 球员移动数据 (2026) | data.mendeley.com/datasets/3sp4xntp34 | 884 段视频（手机加 YouTube），帧级组织 | 标签不明，价值有限 | CC BY 4.0 |

---

## 6. 四个已知问题的具体方案对照

| 问题 | 推荐组合（按投入从小到大） |
|---|---|
| 远端球员小、被网遮挡 | ① 远端 ROI 放大 2× 后做 RTMW-x 或 ViTPose++ 二次推理；② 关键点置信度门控，加时序插值和平滑；③ 击球判定不依赖远端手腕，改用球拍 top 点（RacketVision）；④ 需要 3D 时用 SAM 3D Body，以 2D 点作提示 |
| 双打同色衣服 ID 切换 | ① 阵营约束（同 ID 不跨网）加 ID 上限 2 或 4；② GTA-Link 离线拆分和合并；③ 加入运动学指纹特征（骨长比、持拍手）；④ CAMELTrack 关键点关联；⑤ 疑难片段用 SAM 3.1 掩码兜底 |
| 邻场路人 | homography 脚点落在球场多边形加边距内，再在 tracklet 级别统计在场比例（参考双打论文的 court ROI 做法）；可选：只保留持拍且运动能量高的轨迹 |
| 击球帧 hitter 被遮挡 | 球拍 top 点距离加手腕距离加时间窗 ±k 帧的联合打分；羽毛球轨迹方向翻转点所在侧决定阵营；双打同侧两人之间再按拍头和手腕距离打分，或用 ST-GCN 击球者分类器（参考双打论文） |

---

## 7. 来源
- RTMW: https://arxiv.org/abs/2407.08634 · mmpose: https://github.com/open-mmlab/mmpose
- ViTPose++: https://arxiv.org/abs/2212.04246 · https://github.com/ViTAE-Transformer/ViTPose · https://huggingface.co/usyd-community/vitpose-plus-small
- DETRPose: https://arxiv.org/abs/2506.13027 · https://github.com/SebastianJanampa/DETRPose
- RF-DETR Keypoint: https://rfdetr.roboflow.com/latest/learn/run/keypoints/ · https://github.com/roboflow/rf-detr
- Sapiens2: https://arxiv.org/abs/2604.21681 · https://github.com/facebookresearch/sapiens2 · https://huggingface.co/docs/transformers/model_doc/sapiens2 · Sapiens v1: https://github.com/facebookresearch/sapiens
- GEM-X: https://github.com/NVlabs/gem-x · https://huggingface.co/nvidia/GEM-X · SOMA: https://github.com/NVlabs/SOMA-X · GENMO: https://github.com/NVlabs/GENMO
- SAM 3D Body: https://arxiv.org/abs/2602.15989 · https://github.com/facebookresearch/sam-3d-body · MHR: https://github.com/facebookresearch/MHR
- Fast SAM 3D Body: https://github.com/yangtiming/Fast-SAM-3D-Body · SAM-Body4D: https://arxiv.org/abs/2512.08406 · https://github.com/gaomingqi/sam-body4d
- WHAM https://github.com/yohanshin/WHAM · TRAM https://github.com/yufu-wang/tram · GVHMR https://github.com/zju3dv/GVHMR · PromptHMR https://github.com/yufu-wang/PromptHMR · CameraHMR https://github.com/pixelite1201/CameraHMR · SMPLer-X https://github.com/caizhongang/SMPLer-X · NLF https://github.com/isarandi/nlf · Multi-HMR https://github.com/naver/multi-hmr · 4D-Humans https://github.com/shubham-goel/4D-Humans · MotionBERT https://github.com/Walter0807/MotionBERT · OpenCap Monocular https://github.com/utahmobl/opencap-monocular
- CVBW 2026 workshop papers: https://cvbw2026.github.io/ （论文 11、12、14、15）
- AthletePose3D: https://arxiv.org/abs/2503.07499 · https://github.com/calvinyeungck/AthletePose3D
- GTA-Link: https://arxiv.org/abs/2411.08216 · https://github.com/sjc042/gta-link
- CAMELTrack: https://arxiv.org/abs/2505.01257 · https://github.com/TrackingLaboratory/CAMELTrack
- Deep-EIoU: https://arxiv.org/abs/2306.13074 · https://github.com/hsiangwei0903/Deep-EIoU · MOTIP: https://github.com/MCG-NJU/MOTIP · Hybrid-SORT: https://github.com/ymzis69/HybridSORT · OA-SORT: https://arxiv.org/abs/2603.06034 · SelfTrEncMOT: https://arxiv.org/abs/2510.09878
- SAM 3: https://arxiv.org/abs/2511.16719 · https://github.com/facebookresearch/sam3
- 双打羽毛球: https://arxiv.org/abs/2508.13507 · https://github.com/100-heon/badminton_double_analysis
- 网球发球 3D: https://github.com/jasnwag/tennis_serve_dataset
- RacketVision: https://arxiv.org/abs/2511.17045 · https://github.com/OrcustD/RacketVision · https://huggingface.co/datasets/linfeng302/RacketVision
- CalTennis: https://arxiv.org/abs/2606.20542
- Split-step: https://pmc.ncbi.nlm.nih.gov/articles/PMC11698525/
- VideoBadminton: https://arxiv.org/abs/2403.12385 · FineBadminton: https://arxiv.org/abs/2508.07554 · BFMD: https://arxiv.org/abs/2603.25533 · MultiSenseBadminton: https://www.nature.com/articles/s41597-024-03144-z · Mendeley: https://data.mendeley.com/datasets/3sp4xntp34
