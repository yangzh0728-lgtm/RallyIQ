# RallyIQ 调研：开放许可的业余 / 俱乐部级羽毛球数据集（重点：双打、低机位 / 侧面机位）

> 注：原始调研稿。许可证和数字已另行逐条核查，更正见 `../2026-11-28-web-research-round2.md` 第三节和 `verification.json`；不一致时以核查为准。


> 调研日期：2026-11-28。范围：Kaggle、Zenodo、Roboflow Universe、Hugging Face、figshare、Mendeley、IEEE DataPort、Papers with Code（检索引擎）、GitHub，以及 2025–2026 年随论文发布的数据。
> 已排除任务列表里调研过的条目（TrackNet 系列、ShuttleSet、CoachAI、VideoBadminton、FineBadminton、BFMD、MultiSenseBadminton、BadminSense、kwyoke 等）。
> 许可判定规则：只认 LICENSE 文件、数据集页面的许可字段或论文的明确声明；**没有 LICENSE 文件 = 保留所有权利**；第三方转传时自己挂的许可，不能替代原始权利人的授权。

---

## 0. 结论速览

1. **没有任何一个数据集同时满足"可商用的开放许可 + 业余 / 俱乐部 + 双打 + 低机位 + 击球 / 羽毛球标注"。** 业余双打里标注最好的（Ding 等人的无人机双打数据，有逐帧击球状态和 4 人框）没有许可文件；带开放许可（CC BY 4.0）的业余素材要么没有标注（Mendeley 884 段视频），要么只标了球拍（RacketDB）。专业双打标注（DShuttleNet）是 CC BY-NC-SA，只能做研究和先验。
2. **最接近 RallyIQ 机位的公开数据都不是"开放许可"的：** ETH 机器人视角羽毛球检测数据（接近地面的固定相机，20,510 帧）许可未声明，代码 AGPL-3.0；BMT-BENCH（底线后 2 m、高 4.5 m、60 fps，2,005 个回合片段）仓库没有 LICENSE。两者都要找作者要书面许可。
3. **可以直接放进商用训练管线的只有三类：** RacketDB（CC BY 4.0，业余双打帧，球拍框）、Mendeley 884 段业余视频（CC BY 4.0，无标注，其中一部分来自 YouTube，需要剔除）、Roboflow / Kaggle 上个别标了 CC BY 4.0 的小型检测集（来源不明，需要逐个核对）。
4. **2025–2026 年的新数据集几乎都是"专业 + 非商用"：** DShuttleNet（2026-10，双打，CC BY-NC-SA）、BadmintonSkill-23（WACV 2027，研究用途，门控）、BadmintonGRF（ACM MM 2026，CC BY-NC，8 台 DJI Osmo Action 4）、Fine-Badminton（2026-05，转播单打，CC BY 标签但写明仅限非商业学术）。业余方向的 2026 年工作（单目测速 2611.00031、BadmintonStudio）都没有公开视频数据。
5. **建议的数据策略：** ① 自建业余低机位双打击球集是绕不开的（参考第 4 节的标注规格）；② 用 Mendeley 的 CC BY 素材和 RallyIQ 自有视频做半监督 / 伪标签；③ 给 Ding（无人机双打）、ETH（机器人视角）、BMT-BENCH 三个组发许可申请邮件，拿到后作为评测集；④ DShuttleNet 只用于双打战术先验和 LLM 报告的事实约束，不进可发布模型的权重。

---

## 1. Top 5（按对 RallyIQ 的实际价值排序，已把许可风险算进去）

| # | 名称 | 场景匹配 | 标注 | 数据许可（已核实来源） | 怎么接入 RallyIQ |
|---|---|---|---|---|---|
| 1 | **Ding 等：无人机俯视 + 后视双打数据集**（MTAP 2023） | 业余（大学羽毛球社）男双，39 局、1,347 个回合，后视无人机 4K30 | 4 人框 + 场地坐标；羽毛球逐帧位置 + 状态（飞行 / 击球 / 失误 / 落地 / 误判），12,658 个击球样本 | **仓库无 LICENSE，论文未声明 → 保留所有权利** | 唯一带"击球时刻 + 4 人身份"的业余双打数据。拿到许可后做双打击球检测和击球人归属的评测集；没有许可只能读论文借鉴标注规格 |
| 2 | **Mendeley：Video-Based Frame-Level Dataset of Badminton Player Movements**（2026-06） | 884 段视频，手机 + 数码相机，室内球馆，多角度，比赛和训练 | 无动作标签，只有按视频分好的帧序列 | **CC BY 4.0**（数据集页面） | 剔除 YouTube 来源后，作为无标注业余素材做 TrackNet / RTMO 的伪标签自训练和测试时自适应，补"手机机位"分布 |
| 3 | **RacketDB**（VISAPP 2025） | 同一批业余双打无人机视频的后视帧（27 局、119 回合） | 22,682 张图、16,045 个球拍框（只有 racket 一类） | **Hugging Face 标 cc-by-4.0**；GitHub 仓库无 LICENSE；论文本身 CC BY-NC-ND | 训练球拍检测器，给击球打分器加"球拍–羽毛球距离、拍头速度"特征；在双打里用来判断同侧两人中谁击球 |
| 4 | **DShuttleNet**（arXiv 2610.14016，2026-10） | 专业双打（BWF Super 350+），58 场、4,483 回合、54,141 拍，固定机位 | 每拍击球人、时间戳、羽毛球和 4 人场地坐标、9 类拍型、正反手、击球高度、失分原因 | **CC BY-NC-SA 4.0**（Kaggle API 的 licenseName） | 用作双打先验：击球人序列（同侧两人轮换）、拍型分布、回合长度；给 LLM 报告的双打轮转和防守覆盖做事实约束。不能进商用模型权重 |
| 5 | **ETH：One-Shot Badminton Shuttle Detection for Mobile Robots**（arXiv 2603.06691，2026-03） | 接近地面的固定工业相机，60 fps，1920×1200；5 个室内外场地、11 种背景 | 20,510 帧羽毛球框，分易 / 中 / 难三级 | 代码 **AGPL-3.0**（LICENSE）；**数据许可未声明 → 保留所有权利** | 机位最接近低三脚架。拿到许可后做小目标（<20 px）召回评测；无许可时照搬"半自动标注 + CVAT 修正 + mixup"流程（mixup 召回 0.68→0.78） |

后备：BMT-BENCH（底线后高机位 60 fps 回合片段，无 LICENSE）、Baek & Yun 的"单打模型迁移到双打"研究（量化了双打掉点，代码无 LICENSE）、Roboflow 上标 CC BY 4.0 的球场关键点 / 球员检测小集。

---

## 2. 条目详解

### 2.1 Ding 等：无人机俯视 + 后视双打数据集（Drone_BD_ControlArea）★Top 1

- 论文：Ning Ding 等，*Estimation of control area in badminton doubles with pose information from top and back view drone videos*，Multimedia Tools and Applications（2023），DOI 10.1007/s11042-023-16362-1；arXiv：https://arxiv.org/abs/2305.04247
- 数据 / 代码：https://github.com/Ning-D/Drone_BD_ControlArea （README 给了 Google Drive：`Study_Videos`、`datacsv`、`All_videos`、处理后数据、预训练权重；README 仍写着"The dataset will be fully released soon"）
- 规模与拍摄：大学羽毛球社成员的男双比赛，39 局、14 对、11 名球员、1,347 个回合。两台 DJI Air 2S，一台俯视、一台后视，4K（3840×2160）30 fps。论文没写球员水平和无人机高度。
- 标注：4 名球员的框和场地 XY（ByteTrack 跟踪，人工修正 ID）；羽毛球逐帧 `frame, visibility, x, y, status`，状态五类（论文写作 Frying/Hit/Fault/Drop/Misjudge，Frying 应为 Flying）；TrackNet 预标 + MATLAB GUI 人工修正。README 注明：**只给研究用到的击球 / 落地样本标了羽毛球位置**，落地只标落在对方场地的情况。训练样本 12,658 个击球、796 个落地。
- 许可：**GitHub 仓库没有 LICENSE（gh API 返回无许可），论文正文也没有数据许可 → 按保留所有权利处理。**
- 报告结果：控制区域模型整体 L1 0.094（击球 0.085，落地 0.238）；去掉速度 0.125，去掉姿态 0.110。控制区域占比与得分的 Spearman ρ：按局 0.434，按搭档 0.613。
- **接入 RallyIQ**：这是目前唯一公开的"业余双打 + 逐帧击球状态 + 4 人身份"数据，后视无人机视角介于转播和底线三脚架之间。拿到书面许可后：① 把 `Hit` 状态帧当击球真值，评估 GBM 打分器在双打上的召回；② 用 4 人轨迹训练"同侧两人谁击球"的归属模型（双打的击球人不严格交替，单打的交替约束要改成"按侧交替"）。没有许可时只借鉴它的标注格式（状态五分类正好对应 RallyIQ 的击球 / 落地 / 失误事件）。

### 2.2 Mendeley：A Video-Based Frame-Level Dataset of Badminton Player Movements for Motion Analysis ★Top 2

- 链接：https://data.mendeley.com/datasets/3sp4xntp34 （DOI 10.17632/3sp4xntp34.1，v1，2026-06-22）
- 作者：Suraiya Khandakar、Nashid Nawaz Jisan、Md. Kazi Musfiqur Rahman、Sadaf M Anis（Daffodil International University）
- 内容：884 段视频，**手机和数码相机**在室内球馆拍摄，比赛和训练都有，"不同视角"；另有一部分是从 YouTube 下载的公开比赛 / 训练视频（页面没给数量）。视频已按固定帧率抽成帧、按源视频分文件夹。页面没写帧率、分辨率、单 / 双打、球员水平和总大小。
- 标注：**没有动作标签**，只有"帧级组织"（按视频保持帧序）。页面列出的内容范围是步法、站位、变向和击球动作。
- 许可：**CC BY 4.0**（数据集页面）。没有关联论文。
- 报告精度：无。
- **接入 RallyIQ**：是目前唯一大规模、开放许可、以手机拍摄为主的羽毛球原始素材。用法：① 先人工过一遍，**剔除 YouTube 来源的片段**（上传者无权把别人的视频改成 CC BY）；② 剩下的手机素材作为无标注池，按 lowlabel 报告里的"物理过滤伪标签"配方给 TrackNetV3 和击球打分器做自训练；③ 挑出低机位 / 侧面机位的片段，补进 RTMO 的测试时自适应集。注意：只有帧没有原视频，**音频没有**，不能用于音频击球检测。

### 2.3 RacketDB ★Top 3

- 论文：Muhammad Haq、Shuhei Tarashima、Norio Tagawa，*RacketDB: A Comprehensive Dataset for Badminton Racket Detection*，VISAPP 2025，pp. 426–433，DOI 10.5220/0013159700003912（PDF：https://www.scitepress.org/Papers/2025/131597/131597.pdf ）
- 数据：https://huggingface.co/datasets/muhabdulhaq/racketdb （约 247 GB，跨卷 zip，HF 数据卡为空）；说明页 https://github.com/muhabdulhaq/racketdb
- 来源：Ding 等人的业余男双无人机**后视**视频（4K30），27 局、119 个回合。TrackNet 视频只用于测试泛化，没进训练集。
- 标注：22,682 张图（16,608 训练 / 2,899 验证 / 3,175 测试），16,045 个球拍框，**只有 racket 一类**；被网挡住一部分的球拍不标。CVAT 标注，有 COCO、VOC、YOLO 等七种格式。
- 许可：**Hugging Face 数据集标签为 cc-by-4.0**；GitHub 仓库没有 LICENSE；论文 PDF 本身是 CC BY-NC-ND 4.0（指论文，不是数据）。**风险**：源视频属于 Ding 等人，而那批视频没有许可。RacketDB 作者与 Ding 不是同一团队，CC BY 能否覆盖源帧存疑，商用前应当向两方确认。
- 报告结果：RacketDB 测试集 mAP50 / mAP50:95——YOLOv8 0.78 / 0.47，Faster R-CNN R50 0.78 / 0.47，YOLOv5 0.77 / 0.48，DETR 0.70 / 0.35；COCO 预训练 YOLOv5 的 tennis racket 类只有 0.63 / 0.39。
- **接入 RallyIQ**：① 训练一个轻量球拍检测器（或在 RTMO 的手腕关键点附近做 ROI 检测），给 GBM 击球打分器加"羽毛球到拍框的最小距离""拍框速度峰值""拍框与手腕的相对位置"等特征，专门补低机位下轨迹转折点不明显的击球；② 双打里同侧两人都在挥拍时，用"拍框与羽毛球距离"决定击球人。视角是高位后视，迁移到低三脚架要做数据增强（透视扭曲、缩放）。

### 2.4 DShuttleNet：A Badminton Doubles Dataset for Evaluating Team Coordination ★Top 4

- 论文：Guan-Yi Jhang、Wei-Kai Chen、Hsu-Chao Lai 等（阳明交大、长庚大学、逢甲大学），arXiv 2610.14016（2026-10）：https://arxiv.org/abs/2610.14016
- 数据：Kaggle https://www.kaggle.com/datasets/anonymousay58dki/dshuttlenet （DOI 10.34740/kaggle/dsv/16097938，约 35.7 MB，**只有标注，没有视频**）
- 内容：58 场双打（男双、女双为主，10 场混双），4,483 个回合，54,141 拍；视频来自 BWF 官网 Super 350 以上赛事，固定机位，统一为 30 fps、1280×720；由专业球员人工标注。
- 标注：每拍的击球人和时间戳、正反手、击球高度；羽毛球和 4 名球员的场地坐标（61×134 网格）；9 类拍型（吊、推、杀、抽、挑、网前、高远、发长、发短，由单打 18 类合并）；6 类失分原因并指明责任人；双打特有的相对角度、防守覆盖等字段。
- 许可：**CC BY-NC-SA 4.0**（Kaggle API 返回的 licenseName）。论文未另行声明；没有代码仓库。
- 报告结果：拍型 / 落点预测基线——TSCVAE 拍型准确率 0.504、F1 0.295、FDE 0.524；MoCVAE 准确率 0.466、F1 0.258。论文还报告 TLAI 指标与防守失败率呈非线性关系，阈值约 0.2。
- **接入 RallyIQ**：专业转播机位，不是业余，也没有视频，所以**不能直接训练视觉模型**。可用之处：① 统计双打击球人序列（同侧两人轮换的概率、连续两拍同侧同人的比例），作为 RallyIQ 双打击球人归属的 HMM / DP 先验；② 9 类拍型的双打分布可以给拍型 GBM 做类别先验校正；③ 失分原因六分类和"责任人"字段可以直接作为 LLM 教练报告的双打归因模板和事实检查清单。NC-SA 意味着这些统计量最好只在研究阶段使用，商用版本应当用自有数据重新估计。

### 2.5 ETH：One-Shot Badminton Shuttle Detection for Mobile Robots ★Top 5

- 论文：Florentin Dipner、William Talbot、Turcan Tuna、Andrei Cramariuc、Marco Hutter（ETH Zürich Robotic Systems Lab），arXiv 2603.06691（2026-03）：https://arxiv.org/abs/2603.06691
- 代码：https://github.com/leggedrobotics/shuttle_detection ；项目页 sites.google.com/leggedrobotics.com/shuttlecockfinder ；数据在论文补充表里的 Google Drive（README 的下载链接还是占位符 example.com）
- 内容：20,510 帧，5 个室内外地点、11 种背景；Basler acA1920-144uc（8 mm 镜头）固定拍摄，60 fps，1920×1200，是机器人的视角（低、靠近场地）。定性测试用了腿足机器人上的移动相机。没有写单 / 双打（看描述像单人对练）。
- 标注：羽毛球框，每个框标易 / 中 / 难；去掉了看不到羽毛球的帧；训练只用易 + 中（占 95.9%），另加 1,000 张 COCO 背景图压误检。
- 许可：代码 **AGPL-3.0**（README 说明是因为用了 Ultralytics YOLOv8，已读 LICENSE 段落）。**数据许可在论文和 README 里都没有声明 → 保留所有权利。**
- 报告结果：YOLOv8 在相似环境 F1 0.86，完全陌生环境 F1 0.70；按背景交叉验证平均精确率 0.957；按地点的召回在 0.141–0.892 之间大幅波动；羽毛球小于约 20 px 时召回下降，小于 15 px 时精确率也下降；mixup 把召回从 0.68 提到 0.78；自动标注正确率 85.7%。
- **接入 RallyIQ**：① 它的"按地点召回 0.14–0.89"说明换球馆掉点非常厉害，和 RallyIQ 多场馆的问题同源，可以作为 TrackNetV3 在陌生场馆的压力测试集（需要许可）；② 自动标注 → CVAT 修正 → 分难度的流程可以直接照搬到 RallyIQ 自建标注；③ mixup 和 COCO 背景负样本这两个技巧可以加进 TrackNetV3 的微调。AGPL 代码不要直接并入产品。

### 2.6 BMT-BENCH：A Benchmark Sports Dataset for Video Generation（ICIP 2024）

- 论文：Ziang Shi、Yang Xiao、Da Yan、Min-Te Sun、Wei-Shinn Ku、Bo Hui，ICIP 2024；NSF PAR：https://par.nsf.gov/servlets/purl/10568965
- 数据：Google Drive https://drive.google.com/drive/folders/1moYDb8tp5K-VDxPJU3sTorfYE7NnwVpf ；基线代码 https://github.com/ziangshi/BMT_BENCH_baseline_repo
- 内容：2,005 个从发球到得分的回合片段（无标注子集）；另有一个带羽毛球框的有标注子集（论文没给大小）。相机在底线后约 2 m、高约 4.5 m、俯角 30°，1280×960、60 fps。按得分方（远端红 / 近端蓝）分类，标了回合起止和比分，论文说拍型也在拍级别标注了。论文写"两名球员"，没有明确单 / 双打，也没写球员水平。
- 许可：**论文未声明；基线仓库没有 LICENSE → 保留所有权利。**
- 报告结果（视频生成，8 帧输入预测 12 帧）：IVRNN PSNR 35.81、SSIM 0.9894、LPIPS 0.0062、FVD 967.39。
- **接入 RallyIQ**：底线后固定机位、60 fps、按回合切好、带得分方，正好可以拿来评测 RallyIQ 的回合切分和"谁得分"推断；有标注子集可以测羽毛球检测。机位比 RallyIQ 高约 3 m，属于"偏高的底线后机位"。需要先发邮件要许可。

### 2.7 Baek & Yun：Bridging the Gap: Doubles Badminton Analysis with Singles-Trained Models

- 论文：Seungheon Baek、Jinhyuk Yun（崇实大学），arXiv 2508.13507：https://arxiv.org/abs/2508.13507 ；期刊版 Signal Processing: Image Communication（2026，ScienceDirect S0923596526001888）
- 代码 / 权重：https://github.com/100-heon/badminton_double_analysis （**没有 LICENSE → 保留所有权利**）
- 数据：训练用 ShuttleSet 单打；双打只做验证，来自两场 BWF 决赛转播（2025 瑞士公开赛男双、2022 全英赛女双），逐球员帧标 shot / notshot，男双 1,698 帧、女双 1,276 帧，正负各半。转播俯视机位。
- 报告结果：ST-GCN + Transformer 在双打上准确率 0.6686（P 0.631、R 0.812、F1 0.710），男双 0.702、女双 0.639；单打 0.8617。YOLOv11x 基线双打 0.5941、单打 0.9876。作者称单打→双打掉点 19.3%（基线 39.9%）。双打阈值是在同一个两场验证集上选的，数字偏乐观。
- **接入 RallyIQ**：这是"用单打数据训练的击球模型搬到双打会掉多少"的直接量化，和 RallyIQ 用 ShuttleSet 训练 GBM 的处境一样。可借鉴：按球员独立判断"是否在击球"的骨架时序分类，作为击球候选的第二信号（RTMO 关键点已有）；双打评测必须单独做。数据量太小、没许可，不能拿来训练。

### 2.8 CoachAI Badminton Challenge 2023 Track 1（AI CUP 2023）及其 Zenodo 转传版

- 官方：https://sites.google.com/view/coachai-challenge-2023/tasks/track1 ；数据需填表申请（forms.gle/znfgo4Bvp3t9h8wk9），页面**没有许可或使用条款**。来源是 BWF 国际赛事转播，"从球员后上方拍摄"，30 fps、1280×720，每个回合一个 MP4 加一张逐拍 CSV。
- 标注字段：HitFrame、Hitter（A/B）、BallHeight、RoundHead、Backhand、LandingX/Y、HitterLocationX/Y、DefenderLocationX/Y、BallType（9 类）、Winner。
- 转传：参赛队员 Yu-Hsi Chen 把数据和模型放到 Zenodo https://zenodo.org/records/14677727 （2025-01，assets.zip 14.5 GB：800 段带真值的训练片段、169 段验证、230 段测试，外加 SwingNet / ViT / YOLOv5m 权重），页面标 **CC BY 4.0**。对应代码仓库 wish44165/A-New-Perspective-for-Shuttlecock-Hitting-Event-Detection 没有 LICENSE。
- 许可判断：**参赛者无权把 BWF 转播画面和主办方的标注改成 CC BY，转传版的 CC BY 不可信，按研究用途处理。** CoachAI-Projects 主仓库是 MIT，但那是代码许可，Track 1 的数据不在仓库里。
- 报告结果：该队在 AICUP 2023 排第 11/30（public 0.0727，private 0.0622），CodaLab 决赛 0.3483（2/2）。
- **接入 RallyIQ**：不是业余、不是低机位，但它的 HitFrame + Hitter + 落点 + 双方站位的逐拍格式和 RallyIQ 需要的击球真值格式几乎一样，可以直接作为 RallyIQ 自建标注的字段规格；研究阶段也可以用来预训练击球打分器（只限内部实验）。

### 2.9 BadmintonGRF（ACM MM 2026 数据集赛道）

- 论文：Kuoye Niu 等，arXiv 2605.01876：https://arxiv.org/abs/2605.01876
- 数据 / 代码：Tier 1 在 Zenodo https://doi.org/10.5281/zenodo.19277566 ；代码 https://github.com/KenyaNiu/BadmintonGRF
- 内容：武汉体育学院 17 名国家二级以上运动员（公开基准用 10 人）；**最多 8 台 DJI Osmo Action 4，约 120 fps、1080p**；4 块 Kistler 测力台和 8 相机 Vicon。结构化步法训练和类比赛回合。156 次试验，17,425 个冲击片段，过滤后 12,867 个，1,732 个独立冲击。
- 标注：COCO-17 姿态（YOLO26-pose + ByteTrack）、垂直地面反作用力 Fz、冲击窗口（落地峰 ±0.5 s）、疲劳 / 阶段标签。没有拍型标签。
- 许可：**Tier 1（姿态 + GRF + 元数据）CC BY-NC 4.0**（仓库 LICENSE-DATA 徽章与 README 说明）；**代码 MIT**；Tier 2（原始 RGB、C3D）受控访问，禁止再分发。
- 报告结果：LOSO 单视角 r² 约 0.36–0.40（PatchTST 0.403，ST-GCN+Transformer 0.394）；后期融合可把 TSMixer 提到 0.472；同一试验内 r² 约 0.6。
- **接入 RallyIQ**：相机型号和 RallyIQ 的 DJI 运动相机一致，关键点格式和 RTMO 同为 COCO-17。数据是非商用的，只能在研究阶段验证"从单视角 COCO-17 推断落地冲击 / 步法负荷"能做到多准（r² 约 0.4，偏弱）。MIT 代码里的数据加载和 LOSO 评测框架可以复用。

### 2.10 BadmintonSkill-23（WACV 2027）

- 论文：Naichuan Zheng 等，*BadmintonSkill-23: A Multimodal Benchmark for Fine-grained Badminton Stroke Understanding with Professional Athletes*，WACV 2027
- 代码：https://github.com/zhengnaichuan2022/BadmintonSkill-23 （**MIT**，README 说明）；数据：https://huggingface.co/datasets/NaichuanZheng/BadmintonSkill-23 （门控，HF 许可标签 other）
- 内容：23 名运动员（匿名 S001–S023），RGB + 2D 骨架 + 5 节点 IMU；11 类拍型识别 + 时序动作检测（TAD）；伦理批号 BSU 2026299H。
- 许可：**数据仅限研究、非商业**（README 指向 HF 上的 `documents/03_dataset_license.txt`），RGB 还要单独签 DUA。
- 报告结果：没有核实到具体数字。
- **接入 RallyIQ**：专业运动员、实验室式采集，跟业余低机位相差大。可作为拍型 TAD 的外部评测基准；MIT 代码里的 OpenTAD / PySKL 适配可以参考，用来把 RallyIQ 的"击球检测 + 拍型分类"两段式改成一段式 TAD 的实验。

### 2.11 Fine-Badminton（北航，Zenodo，2026-05）

- 链接：https://zenodo.org/records/20292976 （DOI 10.5281/zenodo.20292976，Fine-Badminton.zip 6.5 GB）
- 作者：Tianyu Wang、Junjie Wu、Jingquan Gao、Shishuo Li（北京航空航天大学）。注意：与已调研的 FineBadminton（iLearn-Lab，ACM MM 2025）不是同一个数据集。
- 内容：10 场 YouTube 公开转播的**单打**，980,703 帧（约 10.72 h），9,817 个帧级时序标注，29 类拍型。
- 许可：页面许可字段是 **CC BY 4.0**，但描述又写明**仅供非商业学术研究**，原视频版权归原所有者。两者矛盾，按非商用处理。
- **接入 RallyIQ**：29 类细粒度拍型体系比 RallyIQ 现有的拍型类别细，可以作为拍型 GBM 扩类时的参考分类表；不能用于商用训练。

### 2.12 Roboflow Universe 羽毛球检测 / 关键点小集（合并为一条）

- 已核实许可的：
  - shuttle-tesmk/badminton_players：**CC BY 4.0**（页面），https://universe.roboflow.com/shuttle-tesmk/badminton_players
  - shuttle-tesmk/badminton_shuttle_label_v4.1：**CC BY 4.0**（页面，2026-04），https://universe.roboflow.com/shuttle-tesmk/badminton_shuttle_label_v4-1 。页面工作流示例里出现 player、shuttle、court_point 三类，没有确认是否就是数据集的类别。
- 未能核实许可的：badyfriends/badminton-shuttlecock-dv7zr（第三方仓库 ZAYN005/badminton-shuttlecock-detection 称其为 Public Domain、3,067 张图，并称在转播和业余比赛画面上都测试过，但 Roboflow 页面抓取没有显示许可）；球场关键点类 badminton-analysis-w9ude/badminton-court-keypoints、tennis-fyp/badminton-court-key-point-1/2/3、gerdos-flow/badminton-calibration（页面抓取没有显示许可和图片数）。
- 共同问题：多数项目没有描述，画面来源（转播截图还是自拍）不明；许可由上传者自填，**上传者不一定拥有画面版权**。
- 报告精度：ZAYN005 用 badyfriends 数据训练 YOLOv8n，mAP50 0.764、召回 0.719（作者自报）。
- **接入 RallyIQ**：价值有限，只适合补球场关键点检测（CourtKeyNet 一类）和球员框检测的少量样本。下载前在页面上逐个截图存档许可字段，只用标了 CC BY 4.0 / Public Domain 并且画面看起来是自拍的项目。

### 2.13 Kaggle 羽毛球素材（合并为一条）

Kaggle API 列出的、和视频 / 视觉有关的条目（许可均为 Kaggle 页面 licenseName 字段）：

| 数据集 | 许可 | 大小 | 内容 | 评价 |
|---|---|---|---|---|
| areebsfatima/badminton（2026-09） | CC BY 4.0 | 278 MB | Roboflow 导出的 YOLO 检测集，图片名为 `video_label_1_*`，无描述 | 来源不明，可作检测补充 |
| shenhuichang/badminton-storke-video（2023） | ODbL（数据库）/ DbCL（内容） | 1.32 GB | 6 类拍型短片（正手抽、挑、网前、高远；反手抽、网前），标签含 beginner | 可作拍型分类的初学者补充样本；ODbL 要求衍生数据库同样开放 |
| zscare/badminton（2025） | Apache 2.0 | 65 MB | 无描述 | 内容不明 |
| phuc25111/shuttlecock-tracknetv2（49.8 GB）、badminton-testset（27 GB） | 页面标 Apache 2.0 | — | 看名字是 TrackNetV2 数据的转传 | **转传者无权改许可**，不可信 |
| nicost312/badminton-footwork、lonavath/badminton-pose、ayushsinha731/shuttle-badminton-photos | Unknown | 0.4–2.2 GB | 步法 / 姿态 / 羽毛球照片 | 无许可，不可用 |

- **接入 RallyIQ**：只有 shenhuichang 的拍型短片（初学者、ODbL）对拍型分类有一点价值，其余基本不可用。

### 2.14 其他（简述，未进主排名）

- **BadmintonQA**（MMSports 2026，https://github.com/Ning-D/BadmintonQA ，无 LICENSE）：19 场 BWF 全场转播（其中 7 场男双）的 533 道四选一问答，视频不再分发。可以参考它的问题分类（识别 / 计数 / 时序 / 定位 / 战术）来设计 RallyIQ LLM 报告的事实核查题。
- **DoublesEval**（https://github.com/Chengjt1999/DoublesEval ，Apache-2.0，代码"即将发布"）：专业双打上的 VLM 战术推理诊断，目前没有数据。
- **TrackNetV2 Shuttlecock Trajectory Dataset 的业余子集**（https://hackmd.io/@TUIK/rJkRW54cU ）：3 段作者自己打的业余比赛、9,525 帧、30 fps 720p，逐帧羽毛球坐标。**页面无许可**。TrackNet 已调研，这里只提示它的业余子集存在。
- **Badminton360-Shuttlecock**（https://huggingface.co/datasets/aissur320/Badminton360-Shuttlecock ，2026-10）：2,758 张图、19.4 GB，YOLO 格式（类 0 和 3 合并为羽毛球），**没有数据卡和许可**。
- **howard9199/Badminton**（https://huggingface.co/datasets/howard9199/Badminton ）：1.64 万条视频、66.3 GB，门控，页面写"正在建设、只对少数人开放"，**没有数据卡和许可**。
- **手机端杀球测速系统**（Diwen Huang，arXiv 2509.05334，2025-09）：YOLOv5 + 卡尔曼滤波的手机 App，面向业余用户；摘要没有提到采集或发布数据，也没有代码链接。
- **单目羽毛球测速**（Huang 等，arXiv 2611.00031，2026-11）：手机拍摄、业余场景，用双手机 240 fps 立体作参考，评估了 51 拍；轨迹和标定"可向作者索取"，原始视频不公开，许可未声明（详见 lowlabel 报告第 13 条）。
- **BadmintonStudio**（https://github.com/balabooooo/BadmintonStudio ，MIT）：业余多场地自动剪辑工具，README 提到测试素材里有 GoPro 拍摄的多场地场馆视频和含双打段落的 82 分钟片段，但**没有公开数据**（详见 rally 报告第 2 条）。
- **跨运动替代**：网球 RallyArb 基准（arXiv 2610.38042）数据 CC BY 4.0，包含手机和场边低机位的击球 / 弹地标注（详见 rally 报告第 3 条），是目前已知唯一"开放许可 + 低机位 + 击球标注"的球拍运动数据，可以给击球检测做跨运动预训练。

---

## 3. 许可汇总

| 条目 | 数据许可 | 代码许可 | 商用训练 |
|---|---|---|---|
| Ding 无人机双打 | 未声明（仓库无 LICENSE）→ 保留所有权利 | 同仓库，无 LICENSE | **不可**，需书面许可 |
| Mendeley 884 段视频 | CC BY 4.0 | — | **可**（署名；剔除 YouTube 片段） |
| RacketDB | HF 标 CC BY 4.0（GitHub 无 LICENSE；论文 CC BY-NC-ND） | — | 表面可，源视频权属存疑，建议确认 |
| DShuttleNet | CC BY-NC-SA 4.0 | 无代码 | **不可** |
| ETH 机器人视角检测 | 未声明 → 保留所有权利 | AGPL-3.0 | **不可**，需许可；代码不要并入产品 |
| BMT-BENCH | 未声明 → 保留所有权利 | 无 LICENSE | **不可**，需许可 |
| Baek & Yun 双打迁移 | 未发布数据 | 无 LICENSE | **不可** |
| CoachAI Track 1 / Zenodo 转传 | 官方无条款；转传版标 CC BY 4.0 但不可信 | 转传代码无 LICENSE；CoachAI 主仓库 MIT | **不可** |
| BadmintonGRF | Tier 1 CC BY-NC 4.0；Tier 2 受控 | MIT | 数据**不可**，代码可 |
| BadmintonSkill-23 | 研究用途、非商业、门控 | MIT | 数据**不可**，代码可 |
| Fine-Badminton | CC BY 4.0 标签 + "仅限非商业学术"声明 | — | 按**不可**处理 |
| Roboflow 小集 | 逐项不同（已核实两项 CC BY 4.0） | — | 逐项核对，来源存疑 |
| Kaggle 小集 | 逐项不同 | — | 大多不可信 |

---

## 4. 对 RallyIQ 的落地建议

1. **自建"业余低机位双打击球集"是主线**，外部数据只能补充。建议标注规格直接合并 CoachAI Track 1 的逐拍字段（HitFrame、Hitter、站位、落点、拍型、Winner）和 Ding 数据集的逐帧羽毛球状态（Flying / Hit / Fault / Drop），再加双打必须的"同侧两人中谁击球"（A1/A2/B1/B2）。先标 20–30 段（每段一局），按场馆和机位（底线后低位、侧面低位）分层，这样 34/58 这类召回数字才有统计意义。
2. **发三封许可申请邮件**（Ning Ding 等人的无人机双打、ETH RSL 的机器人视角数据、Ziang Shi 等人的 BMT-BENCH），说明是内部评测用途，不再分发。拿到后作为固定的外部评测集，不进训练。
3. **开放许可素材用于半监督**：Mendeley CC BY 素材（剔除 YouTube 片段）加 RallyIQ 自有视频，用 TrackNetV3 + 物理残差过滤生成伪标签；RacketDB 训练球拍检测器，给击球打分器加球拍特征。
4. **双打先验只在研究阶段用 DShuttleNet**：先用它估计"同侧轮换"的转移概率，验证双打击球人归属的 DP 是否有效；商用版本改用自建数据重新估计。
5. **单打→双打必须单独评测**：Baek & Yun 报告单打训练的击球模型在双打上掉 19–40%，RallyIQ 的 GBM 打分器很可能类似，双打视频要有独立的召回 / 精度看板。

---

## 5. 局限与未核实项

- 本次调研中途 WebFetch 配额（多代理共享、每小时 100 次）多次耗尽；curl 只能访问 GitHub，Kaggle、Zenodo、Mendeley、Hugging Face、arXiv、Roboflow 都只能通过 WebFetch 或 Kaggle API 读取。
- **未能核实**：arXiv 2308.04872（双相机球场球员跟踪）是否附带数据；arXiv 2509.05334 只读了摘要；BadmintonSkill-23 的具体精度；Roboflow 多数项目的许可和图片数；HF 上 Tinghong-Ye66/badminton_dataset（MIT）、Ngoctan123/Badminton（MIT）、JIAWEIII/badminton 的具体内容；Mendeley 数据集中 YouTube 片段的数量和总大小；BMT-BENCH 的球员水平和单 / 双打。
- 精度数字都是作者自报，且多数基于转播或实验室场景，不能当作 RallyIQ 业余低机位上的预期性能。
- 许可判断不是法律意见；任何"表面可商用"的条目（RacketDB、Mendeley、Roboflow），商用前都应保存许可页面快照，并核对源画面的权属。
