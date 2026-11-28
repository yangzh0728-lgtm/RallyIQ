# 按训练线程整理的可执行建议（2026-11-28 网上调研）

基于 RallyIQ main 503aea2 的代码核查。详细依据见 `../2026-11-28-web-research.md` 和 `code-mapping.md`。许可证红线：GPL/AGPL/非商用/无许可证的代码和权重只借思路或仅内部评测。

## 击球检测（Keep adding training data and improve hit detection）
1. 最大杠杆（PLAN 步骤 B 已点名）：`events/learned.py:89-92` 的 dir_in/out_x/y 是图像单位向量。改成相对击球人或相对球网方向（不需要 H），或者用 H 投到球场平面（`ShuttleObs.court_xy` 目前从未填充）。依据：Beyond the Premier（CVPRW 2024）——转播训练的事件检测在单机位非职业画面上紧 mAP 61.8→19.6；相机无关的射线特征（Where Is The Ball，CVPRW 2025）能泛化。
2. 挥拍证据：Hsu 等（Sensors 2024）规则“一个挥拍段最多一个击球、没挥拍不算击球”，精度 0.588→0.897，F1 0.723→0.905（召回 0.936→0.913）。RallyIQ 的 `pose_hits.swing_curve` 已有但默认关闭且只补漏。建议把腕速 / p_swing 作为打分器特征（软约束，远端球员 DJI 上约 23% 帧缺失），并在 `hits.py:177-188` 按挥拍段合并。
3. 后处理：先对每个候选打分，再高斯平滑（σ≈3）+ 峰值检测（lRomul，SoccerNet BAS 2023 冠军，MIT），需在贪心合并之前打分。
4. 第二个业余测试集：GitHub kwyoke/Badminton-hit-detection，业余单打 + 业余双打、底线机位、击球帧标注（Google Drive）。无许可证，仅内部评测。
5. 以后的大项：AdaSpot（CVPR 2026，MIT，HF 权重）或 E2E-Spot（BSD-3）训练“击球”单类 RGB 检测作第二候选源；或按论文复现 UMEG-Net（AAAI 2026，姿态+球+场角图网络，ShuttleSet 小样本事件 F1 49.2 vs E2E-Spot 42.7，仓库无许可证）。不要用 T-DEED（GPL-3.0）。

## 羽毛球跟踪（Train shuttle tracking model）
1. 标注规则（零成本）：点拖影的中点，不点前沿。BlurBall（CVPRW 2026，MIT）只改这一点，TrackNetV3 F1 93.74→95.93（乒乓球数据）。现在 `label_tool.py` 和 `shuttle-finetune-guide.md` 写的是“点在球上”。
2. 微调加遮挡增强（TOTNet，CVIU 2026，MIT）：随机用人体/球拍块遮住球；完全遮挡帧准确率 0.63→0.80。接入 `training.py:386-396`。按可见性加权的损失需要先在 `evaluation/schema.py` 里把 visible 从 bool 扩成 可见/遮挡/出画。
3. 代码里发现的小不一致：训练每段标注 clip 取一个中值背景（`training.py:137`），推理按机位片段取。应对齐。
4. 伪标签：WASB（nttcom/WASB-SBDT，BMVC 2023，MIT，Drive 上有羽毛球权重，F1 93.1）是结构不同的第二检测器，只在它与 TrackNetV3 一致时采纳自动标签。现在 auto_label 比较的是同一网络的两个版本，错误相关。
5. RacketVision（AAAI 2026）：中值背景 + 4 帧 + 多项目联合训练让 TrackNetV3 羽毛球 mAP@50 57.8→83.1；其球拍框可用于屏蔽手臂/球拍误检。代码 MIT，HF 权重无许可证，仅内部评估。
不采用：TrackNetV5（两个版本均无许可证或专有，比 V3 提升不到 1 个点）；ETH shuttle_detection（AGPL）。

## 球员跟踪与击球人（Train player tracking and hitter ID）
1. 球拍关键点：RacketVision（OrcustD/RacketVision，代码 MIT）羽毛球拍 5 点姿态 PCK@0.2 88.5（多项目模型，单项目 82.1）。击球人以拍头为主、手腕为辅；现在 `events/players.py:81-100` 的拍头是 腕+0.7×(腕−肘) 猜的。HF 权重无许可证、训练画面是 YouTube 转播 → 先内部评估，上线前用自家帧重训。
2. 离线 tracklet 拆分/合并：GTA-Link（sjc042/gta-link，MIT），SportsMOT IDF1 79.8→86.5。加我们的先验：ID 不过网、每侧最多 2 人。接入 `perception/players/runner.py:174-177`。现在缺：段内拆分、搭档互换（P1↔P3）修复；外观只有 HSV 直方图。它默认的 ReID 权重（Deep-EIoU 的）无许可证，需要自己的或别的嵌入模型。
3. 同色球衣时的平局打破：只用姿态的“运动学指纹”（骨长比、持拍手、站宽）。网球发球数据（CC BY 4.0）在 14 人闭集上 99.2%，当作特征，不当识别器。
4. 用关键点参与关联的 tracker：CAMELTrack（Apache-2.0，HF 权重 Apache-2.0），SportsMOT HOTA 80.4，需在双打片段上训练。
注意：双打论文 Bridging the Gap（arXiv 2508.13507）常被引用的 0.6686 其实是“是否击球”帧准确率，不是击球人准确率；仓库无许可证。球场 ROI 过滤路人 RallyIQ 已做。

## 姿态与步法（Train player pose and footwork）
1. 远端球员精修：`perception/players/pose.py:266-267` 的远端二次推理只“补漏人”，已检到但关键点差的不会被改善。对远端/小框做 top-down RTMW-x 384×288（mmpose，Apache-2.0，直接下 .pth），133 点全身含脚跟、大脚趾、小脚趾（COCO-WholeBody 足部 AP 79.6），split-step 离地/落地和触地点不再靠“脚踝 + 框高偏移”。备选 ViTPose++（HF transformers，权重 Apache-2.0）。
2. split-step 参照：顶级男单在对手击球后约 0.24–0.25 s 离地（30fps 约 7–8 帧），站宽约身高 0.5 倍（PMC 2024–25，3 人）。只作展示参照，不作硬阈值。
3. 如果要做 3D 技术分析：GEM-X（NVlabs，代码 Apache-2.0，权重 NVIDIA Open Model License 可商用）或 SAM 3D Body（SAM License，可用 2D 关键点作提示）。不要 SMPL 系（WHAM/GVHMR/PromptHMR/SMPLer-X 非商用）和 Sapiens（v1 非商用，v2 禁止生物特征处理）。评估看关节角和时相，不看 MPJPE；单目角速度不可靠。
4. 触地时序真值：MultiSenseBadminton（Sci Data 2024）有足底压力鞋垫 + 视频，数据许可证未核实。

## 击球类型（Train stroke type classifier）
1. 便宜修正：学习模型在回合中间预测出“发球”时，标记为“可能是合并的回合”（PLAN 步骤 C 的已知问题），而不只是屏蔽。接入 `strokes/classify.py:76-84`；规则后端已禁止中间发球。
2. 前后拍上下文：TemPose-TF-ASF（arXiv 2605.02558）ShuttleSet 31 类 0.835→0.854。`strokes/README.md:151` 记录 RallyIQ 试过一次（60.9% vs 63.6%），未记录是否 out-of-fold。值得在 v3 上用严格 OOF 重试一次。
3. 机位无关：`strokes/context.py:52-56` 的 out_vx/out_vy/in_vy 是图像方向特征，与击球检测同一问题，改成相对击球人或球场平面方向。
4. 区分同一位置出手的吊/杀/高远的视频分支：冻结 V-JEPA 2（代码和权重 MIT）或 VideoMAE V2（权重已公开，Apache-2.0）提击球前后 16 帧特征，接注意力探针，与 GBM 融合。V-JEPA 2 ViT-L 冻结探针在 Diving-48 上 89.0%。
5. 数据：VideoBadminton（qilimk，18 类，7,822 段，底线后 4.5 m 高机位）最接近我们的机位，但无许可证，仅研究/评测。BadminSense（业余、专家质量评分）CC BY-NC-ND，仅内部评测。保留 BST 的话，骨架分支可试 ProtoGCN（CVPR 2025，MIT）。

## 球场检测（Train automatic court line detection）
1. 代码里发现的快速修正：网页手动四角路径（`api/app.py:476`）调用 homography_from_corners，跳过了 `calibrate_from_points(..., image=frame)` 的线吸附精修，改为走后者。
2. CourtKeyNet（adithyanraj03/CourtKeyNet，代码 MIT；HF Cracked-ANJ/CourtKeyNet 权重 MIT）是可学习的羽毛球四角检测器，输出三种置信度：热图峰值、熵、四边形几何合法性。与直线法互为候选；两者不一致或置信度低时主动请用户手动标四角（现在只能用户自己触发）。99.99% PCK 是自报、偏俯视画面，要在我们的侧机位上实测。数据衍生自 Roboflow CC BY 4.0，需署名。
3. 镜头畸变（calibration README 已记未校正）：TVCalib（WACV 2023，MIT）优化线段重投影误差，可选畸变项；DeepLSD（MIT）在低对比场线上比 Hough 好。PnLCalib/NBJW 的点+线精修是 GPL，只借思路。
4. 训练数据：Roboflow BadmintonC（CC BY 4.0），最新版 2,536 张，标注可能是框不是角点。Good-Badminton（Apache-2.0）2026-06 加了球场模板匹配，可在我们的侧角度帧上作对照基线。

---

# 第二轮补充（音频、回合切分、少标注、战术、数据）

详见 `../2026-11-28-web-research-round2.md`。

## 击球检测（补充）
6. 音频候选（无需训练，librosa ISC + scipy BSD）：零相位带通 1.5–10 kHz → 10 ms 能量 / SuperFlux → 分块“中位数 + k×MAD”阈值 → 120–250 ms 不应期。每个视频用高置信视觉击球自动估计音画偏移（含声速延迟 d/343），然后把“最近音频起始点的时间差 / 是否匹配 / 信噪比”作为打分器特征。依据：Huang 等 AVSP 2011 网球双打，视频 F 67.6–70.6 → 融合 77.0–84.2，主要提升精度。串音：回合间歇期的起始点作负例，自监督训练“本场 vs 其他”分类器（PANNs 代码 MIT、权重 CC BY 4.0）。
7. 只靠姿态的击球时刻：腕/肘/肩速度峰值按 0.5/0.3/0.2 加权（arXiv 2606.15992，网球底线机位，±100 ms F1 85.65%），可对照现有 `pose_hits.swing_curve`。
8. 主动学习：按片段计算击球分数熵，先把最不确定的回合给 sam 看（足球上 16% 数据达到全量 90%，随机要 45%）。
9. 已确认 `hits.py:_alternating_subset` 已做 MonoTrack 式交替 DP；MonoTrack 另有“D 秒回合约 D 拍”的先验，可考虑加入。

## 羽毛球跟踪（补充）
6. 测试时自适应：每个新视频重估 TrackNetV3 的 BatchNorm 统计量（TENT 最简版，MIT，零标注），在侧机位视频上 A/B。
7. 自动标签物理过滤（TT4D 思路）：重投影误差、符合带阻力的抛物轨迹、事件顺序合法，三者都过才收为训练标签。羽毛球参数：气动长度约 4.6 m（羽毛片 4.04 m，塑料 4.48 m），Cd 0.65–0.68，终端速度约 6.7 m/s（Cohen 等 2015）。

## 击球类型（补充）
6. 发球站位锚定（移植 Smashers 规则逻辑，MIT，不带它的 AGPL YOLO）：发球人在发球区内静止、接发人在对角区 ≥0.4 s → 发球锚点；长回合内出现发球站位就切开；锚点后第一拍强制为发球。BadmintonStudio 实测：回合内两拍间隔 p99 3.25 s，回合之间通常 ≥4 s。这条同样适合击球检测线程（回合切分在 `events/rallies.py`，现在只看球是否在动）。

## 姿态与步法（补充）
5. 报告里可加的步法指标（BadmintonVision，MIT）：路径效率、回中时间、中场占有率、变向次数。

## 球员跟踪与击球人（补充）
5. 双打基准：DShuttleNet（58 场 BWF 双打，击球人 + 4 人坐标，CC BY-NC-SA）只能内部评测击球人判定。RacketDB（HF 标 CC BY 4.0，22,682 帧球拍框，无人机后视）可用于训练球拍检测。
