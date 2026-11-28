# RallyIQ 少标注学习调研：自训练 / 伪标签 / 主动学习 / 测试时自适应 / 物理仿真合成数据

> 注：原始调研稿。许可证和数字已另行逐条核查，更正见 `../2026-11-28-web-research-round2.md` 第三节和 `verification.json`；不一致时以核查为准。


调研日期：2026-11-28。目标场景：业余羽毛球视频（手机或 DJI 运动相机，低三脚架，放在底线后或侧面，30-60fps，单打和双打，多场地球馆）。
现有栈：TrackNetV3（已微调）+ RTMO（COCO-17）+ 轨迹转折点候选 + GBM 击球打分器（用 ShuttleSet 转播数据训练，在低机位业余视频上 58 个击球只找到 34 个）+ 球场单应 + GBM 球种模型 + 带接地检查的 LLM 教练报告。
本文不重复已调研过的条目（TrackNet V1-V5、WASB、BlurBall、TOTNet、E2E-Spot、T-DEED、UMEG-Net、F3Set、AdaSpot、ShuttleSet、Where Is The Ball、shuttle-speed-cam EKF 等）。

> **许可说明**：每条都写明实际读到的 LICENSE 文件、GitHub API 的 license 字段或数据集页面。"无 LICENSE 文件" = 默认保留所有权利，商用不能直接用。"未声明 / 未能核实" = 没有读到权威许可文本，不做猜测，一律按保留所有权利处理。
> **数字来源**：数字取自论文正文或表格（arXiv 全文 / HTML / CVF PDF）、项目页，或仓库代码与 README，括号里注明出处。

---

## 一、结论先行（TL;DR）

1. **最值得先做的是物理仿真合成数据管线：阻力 ODE 生成轨迹 → 投影到随机的低机位相机 → 加入带噪声的 2D 检测 → 训练击球检测和落点模型。** 乒乓球的 Kienzle 等人（CVPRW 2025 最佳论文，后续 WACV 2026）**训练时完全不用真实标签**，只靠物理正确的仿真加针对性的检测噪声增强，就零样本迁移到真实转播视频（旋转分类 92%，重投影误差为图像对角线的 0.19%）。基于同一合成训练后端的 TT4D 在 21 万多段真实回合（140 多小时）上自动打标，球拍击球时刻 F1 为 0.83。羽毛球的阻力模型只有一个参数：dv/dt = g − C‖v‖v。多个独立来源给出 C ≈ 0.20-0.29 m⁻¹（即气动长度 L = 1/C ≈ 3.4-5 m，终端速度约 6.7-7.0 m/s），可以直接拿来随机采样。
2. **低机位击球召回低（34/58），用"物理约束过滤的伪标签 + 主动学习"比继续堆 ShuttleSet 更划算。** MonoTrack 只加了击球交替、最小间隔和回合时长这几条 DP 约束，就把 HitNet 的 F1 从 0.866 提到 0.946，不需要任何新标签。TT4D 用"重投影误差 + ODE 拟合残差 + 事件合理性"三道过滤筛自动标注，71 万段剪辑最后保留 21 万段。SST 的双阈值教师-学生配方在只有 1% 标签时把检测 mAP 从 18.1 提到 26.2，代码是 BSD-3。Giancola 的熵主动学习只用 16% 数据就达到全量性能的 90%，随机采样则需要 45%。
3. **测试时自适应有零成本的起点**：TrackNetV3 主干用的是 `nn.BatchNorm2d`（已读代码确认），可以对每段视频（同一场地、同一机位）重新估计 BN 统计量。在 TENT 仓库的 CIFAR-10-C 示例里，只重估 BN 就把平均错误率从 43.5% 降到 20.4%，再加熵最小化降到 18.6%。羽毛球热图没有分类熵可用，自监督信号建议改用"检测点到阻力 ODE 拟合曲线的距离"。
4. **许可**：最有价值的几个参考实现（Kienzle 用 GPL-3.0，MonoTrack 用 Adobe Research 非商业许可，TT4D 用 CC BY-NC-SA 4.0，CoTracker3 用 CC BY-NC）都不能直接放进商用代码，**只能借鉴配方、自己重写**。可以直接用的代码是 SST（BSD-3）、TENT（MIT）、CoTTA（MIT）、TTP（MIT）、PPAL（Apache-2.0）、FiftyOne（Apache-2.0）和 Lightly（MIT）。

---

## 二、Top 5 排名（按对 RallyIQ 的投入产出比）

| 排名 | 条目 | 类型 | 许可（代码 / 权重 / 数据） | 关键数字 | 接入 RallyIQ 的方式 |
|---|---|---|---|---|---|
| 1 | Kienzle 等：乒乓球物理合成到真实的迁移（CVPRW 2025）+ Uplifting Table Tennis（WACV 2026） | 物理仿真合成数据 | 代码 GPL-3.0；数据和权重托管在 Augsburg 大学 mediastore，未声明许可 | 零真实标签；旋转分类 92%；2D 重投影误差为对角线的 0.19% | 照搬"随机相机 + 检测噪声增强"的配方，换成羽毛球阻力 ODE，训练只吃 2D 轨迹的击球检测器（自己重写） |
| 2 | MonoTrack（CVPRW 2022） | 羽毛球阻力模型 + 击球约束 DP | Adobe Research License（仅限非商业） | HitNet F1 0.866，加 DP 后 0.946；C 初值 0.2 m⁻¹，范围 [0, 0.4] | DP 约束用作击球后处理和伪标签过滤；在两个击球之间做 ODE 拟合，残差陡增处补漏检的击球（自己重写） |
| 3 | TT4D（ACM MM 2026） | 大规模自动打标 + 物理过滤 | CC BY-NC-SA 4.0（其中 Kienzle 抬升模块为 GPL-3.0） | 71.5 万段剪辑 → 21.2 万段通过；击球时刻 F1 0.83，弹地 F1 0.92 | 照搬"三道过滤器 + 逐阶段通过率统计"，把无标签业余视频批量转成训练数据 |
| 4 | SST：足球球员和球检测的半监督训练（CVPRW 2022） | 教师-学生伪标签 | 代码 BSD-3-Clause；权重未声明；SoccerNet-v3 需签协议 | 1% 标签：18.1 → 26.2 mAP；10% 标签：39.5 → 45.1（两轮） | 把双阈值 τl=0.9、τh=1.0（中间区域忽略）和"学生最后在真标签上微调"用到 TrackNet 热图和击球候选上 |
| 5 | Giancola 等：足球动作定位的主动学习（CVPRW 2023） | 主动学习 | 没有公开代码 | 熵采样用 16% 数据达到全量性能的 90%（随机采样需 45%） | 按片段用熵（片段内取 max）挑最难的业余视频片段送标，预算按 1% → 2% → 5% → 10% 递增 |

第 6-8 名：羽毛球空气动力学参数汇总（第 12 条）、TENT / BN 重估（第 7 条）、TFD 时序特征蒸馏（第 6 条）。

---

## 三、条目详解

### 1. Kienzle 等：物理正确的合成数据到真实转播视频（乒乓球，CVPRW 2025 最佳论文）及后续 Uplifting Table Tennis（WACV 2026）

- 论文：<https://arxiv.org/abs/2504.19863>（CVPRW 2025）；<https://arxiv.org/abs/2511.20250>（WACV 2026）
- 代码：<https://github.com/KieDani/SpinAndTrajectoryTableTennis>；<https://github.com/KieDani/UpliftingTableTennis>（支持 torch.hub）
- 项目页：<https://kiedani.github.io/CVPRW2025/>；<https://kiedani.github.io/WACV2026/>
- **许可**：两个仓库的 LICENSE.txt 都是 **GPL-3.0**（已读；GitHub API 也是 GPL-3.0）。合成数据集、权重和 TTHQ 标注托管在 mediastore.rz.uni-augsburg.de，项目页只写了"Dataset and Weights (see manual on GitHub)"，**没有声明数据和权重的许可**。TTHQ 的视频来自 YouTube。
- **报告结果**：CVPRW 项目页原文是 "No real data at all is required for training"，旋转分类准确率 **92%**，2D 重投影误差为 **图像对角线的 0.19%**。WACV 版把系统拆成前端和后端：前端是 Segformer++ 做的球和球台关键点检测，用新的 TTHQ 数据集监督训练，并用"跨模型过滤"去掉鞋、球拍之类的误检；后端是 2D 到 3D 的抬升网络，**只用合成数据训练**，用"按真实时间戳编码的 RoPE"处理变帧率和漏检，实现零样本泛化。
- **可直接照抄的配方**（从 UpliftingTableTennis 代码读到）：
  - 随机相机（`uplifting/data.py::sample_camera`）：到原点距离 7-17 m；俯仰 θ 在 30-70°；方位角 φ 覆盖约半圈；焦距 fx、fy 在基准值的 0.6-2.0 倍；注视点在台面中心 ±0.2 m 内随机；投影后检查轨迹是否在画面内。
  - 帧率：训练时每条轨迹在 20-65fps 中随机取一个，评测用 50fps（`fps_bounds = (20, 65)`）。
  - 检测噪声（`uplifting/config.py`）：`randomize_std = 8` 像素高斯抖动；`randdet_prob = 0.05` 随机误检；`randmiss_prob = 0.05` 漏检；`tablemiss_prob = 0.05` 球台关键点缺失；`stop_prob = 0.5` 在弹地后截断；`MotionBlur` 在相邻帧之间沿真实轨迹随机取一个位置，模拟运动模糊造成的检测偏移，blur_strength 取 [0.1, 0.5)。
  - 代码里的 `_count_hits` 直接从仿真的 3D 轨迹算出击球和弹地时刻，作为免费的事件标签。
- **接入 RallyIQ**：把 MuJoCo 乒乓球仿真换成羽毛球阻力 ODE（参数见第 12 条）。相机分布改成"低三脚架，底线后或侧面"（高度 0.4-1.6 m，距离 1-12 m，广角加畸变），生成带击球时刻标签的海量 2D 轨迹。用它训练一个只吃"2D 球轨迹 + 球场关键点"的击球检测器，把输出概率作为特征接进现有 GBM；或者直接给 GBM 补低机位的正负样本。GPL 代码只作参考，需要自己重写，避免传染商用代码。

### 2. MonoTrack：从单目羽毛球视频重建穿梭球轨迹（Liu 等，CVPRW 2022）

- 论文：<https://arxiv.org/abs/2204.01899>（CVF：<https://openaccess.thecvf.com/content/CVPR2022W/CVSports/papers/Liu_MonoTrack_Shuttle_Trajectory_Reconstruction_From_Monocular_Badminton_Video_CVPRW_2022_paper.pdf>）
- 代码：<https://github.com/jhwang7628/monotrack>
- **许可**：LICENSE 为 **Adobe Research License**（已读），原文 "noncommercial research purposes ... only. Noncommercial research purposes do not include ... development of commercial products"。商用 RallyIQ **不能用其代码和权重**，只能借鉴论文思路。训练数据基于 TrackNetV2 的公开转播数据集。
- **报告结果**：穿梭球跟踪准确率 88.6%（TrackNetV2 为 84.0%）。击球检测 HitNet 是一个 GRU，输入为球场四角、双方姿态和球的位置：只用球时 F1 0.815，加上姿态 0.850，全部输入 0.866；**再加 DP 约束优化后，召回 94.3%、精度 94.9%、F1 0.946**（基于导数的基线 F1 只有 0.699）。约束有三条：两次击球间隔 ≥ 0.5 s；击球在两侧之间交替；D 秒长的回合大约有 D 次击球。论文还用 1 万条物理仿真的 3D 轨迹评估 3D 重建。
- **物理模型**（从 `python/ai-badminton/src/ai_badminton/rally_reconstructor.py` 读到）：d²x/dt² = g − C‖v‖v。`Cg = 0.2` 是 C 的初值，C 逐拍拟合，范围 `(0, 0.4)`。每帧积分 `substeps = 30` 个子步。速度分量边界为 ±150 m/s。拟合变量还包括起点位置（x ∈ [0, 6.1]，y ∈ [0, 13.4]，z ∈ [0.1, 6]）。
- **接入 RallyIQ**：
  - (a) 把"两侧交替 + 最小间隔 0.5 s + 回合时长先验"的 DP 写进击球后处理，同时用作自训练时的伪标签一致性过滤器。双打时"交替"指的是两侧半场交替，不是固定某个球员。
  - (b) 在每两个相邻的候选击球之间拟合一次阻力 ODE；**拟合残差陡增的地方就是漏掉的击球**，正好补低机位转折点检测漏掉的击球。
  - (c) 同一个 ODE 用来生成第 1 条配方里的合成轨迹。
  - 注意：低机位侧拍时只用地面点做 DLT 会退化，需要再加网柱顶或网带高度的点。

### 3. TT4D：从单目视频做乒乓球 4D 重建的管线和数据集（Rahmanian、Kienzle、Gossard 等，ACM MM 2026）

- 论文：<https://arxiv.org/abs/2605.01234>；项目页 <https://nimsi123.github.io/tt4d/>
- 代码：<https://github.com/Nimsi123/tt4d>
- **许可**：LICENSE 为 **CC BY-NC-SA 4.0**（已读，README 写明 "Except where otherwise noted"）。仓库内置的 Kienzle 抬升模型是 GPLv3，TrackNetV3、TT3D 和 4DHumans 各自沿用原许可。数据集（`tt4d_pickles.zip`，57 GB）托管在 berkeley.edu，没有单独声明许可，来源是 `video_links.txt` 列出的 3581 个 YouTube 视频。**整体不可商用。**
- **方法**：一条 12 阶段的自动管线：计分板 OCR → 按回合剪辑 → 相机标定 → TrackNetV3 2D 跟踪（不用 InpaintNet）→ 4DHumans 人体 → 部分重建 → 用合成数据训练的抬升网络得到 3D 轨迹和旋转 → SAM3 球拍分割。抬升网络训练时的遮挡增强是"经验遮挡"：先统计真实转播里连续可见段和连续遮挡段的长度分布，再交替采样生成遮挡掩码。
- **自动标注的过滤器（可以直接照抄）**：
  - 重投影：最大归一化误差超过球台对角线的 20% 就拒绝；
  - 物理一致性：预测轨迹与 ODE 拟合的最大欧氏距离超过 30 cm 就丢弃；
  - 事件合理性：每段只能弹地一次（发球段两次），至少 2 次击球，峰到峰最短 0.2 s，击球极值点离网至少 0.3 m；
  - 球平均可见率 ≥ 50%，至少两个重建段，人体位置在合理范围内。
  - **通过率**（论文 Table 1）：714,664 段剪辑 → 405,769 段通过比赛检测（56.8%）→ 371,733 段通过标定（91.6%）→ 211,534 段通过重建（56.9%）。重建阶段的拒绝原因依次为：人体位置异常 67,442，段数不足 47,823，重投影误差高 30,937，ODE 残差高 9,736，球可见率低 4,261。
- **报告结果**：时间分段的弹地 F1 为 **0.92**，**球拍击球 F1 为 0.83**。侧视单段 3D 误差 15.04 cm（LATTE-MV 15.27 cm，TT3D 29.86 cm）。抬升网络在合成测试集上的 3D 误差为 2.60 ± 1.15 cm；加入半帧率和漏检后为 3.55 ± 1.63 cm，事件 Macro F1 0.941。
- **接入 RallyIQ**：这是"无标签视频 → 物理过滤 → 伪标签数据集"的完整工业化样板。RallyIQ 可以按同样结构，对所有用户上传的业余视频跑现有管线，再用三道过滤器（重投影残差、阻力 ODE 残差、击球事件合理性）筛出高置信回合，作为 GBM 和 TrackNet 的伪标签。同时记录每个过滤器的拒绝数，用来定位管线的薄弱环节。代码是 NC-SA 许可，只借鉴设计。

### 4. SST：用半监督训练提升足球球员和球检测（Vandeghen, Cioppa, Van Droogenbroeck, CVPRW 2022）

- 论文：<https://arxiv.org/abs/2204.06859>（DOI 10.1109/CVPRW56347.2022.00392）
- 代码：<https://github.com/rvandeghen/SST>
- **许可**：代码 **BSD-3-Clause**（已读 LICENSE，GitHub API 一致，可商用）。预训练权重放在 Google Drive，没有单独声明许可。训练数据 SoccerNet-v3 需要签使用协议（README："We assume that you have the rights to use SoccerNetv3"）。
- **方法**：用部分有标签的比赛训练教师 → 对无标签比赛生成伪标签 → 双阈值过滤（`src/threshold.py`）：分数 ≥ τh 的作为正样本；τl ≤ 分数 < τh 的框标为"忽略"，不参与损失；更低的当背景。论文比较了三种参数化，第 3 种是"双阈值 + 渐进式怀疑"，最好的组合是 τl = 0.9、τh = 1.0。然后学生在"真标签 + 伪标签"上训练，**最后在真标签上微调**；可以迭代第二轮。
- **报告结果**（论文 Table 1，验证集，6 类合并的 mAP（AP50:95））：
  - 1% 标签：教师 18.1 → 学生 26.2（第 3 种参数化）；
  - 5% 标签：31.9 → 38.9；
  - 10% 标签：39.5 → 43.7（第一轮）→ **45.1**（第二轮，含微调）；
  - 100% 标签：52.7 → 53.8；
  - 测试集：教师 51.0 → 学生 **52.3**（用约 160 万帧无标签数据）。
- **接入 RallyIQ**：TrackNetV3 热图版的做法如下。
  - 教师在现有标注和 ShuttleSet 上训练，对无标签业余视频推理；
  - 峰值 ≥ τh 且通过物理一致性（离前后帧 ODE 拟合曲线 < k 像素）的帧当正样本；
  - 峰值在 [τl, τh) 的帧在损失里屏蔽；
  - 低于 τl 且 InpaintNet 也认为不可见的帧当负样本；
  - 学生训练完，最后在少量真实低机位标注上微调。
  - 击球 GBM 的伪标签同理：只有 GBM 分数和 DP 约束一致的才采纳，分数在中间区间的不进训练。

### 5. Giancola 等：足球视频动作定位的主动学习（CVPRW 2023）

- 论文：<https://arxiv.org/abs/2304.04220>（CVF PDF：<https://openaccess.thecvf.com/content/CVPR2023W/CVSports/papers/Giancola_Towards_Active_Learning_for_Action_Spotting_in_Association_Football_Videos_CVPRW_2023_paper.pdf>）
- 代码：论文和 arXiv 页面都没有给代码链接，也没有找到公开实现。
- **许可**：没有可用代码。
- **方法**：把视频切成固定长度、不重叠的片段，以片段为单位挑选。比较了三种采集函数：随机（RS）、不确定性（UM）和熵（EM），**在 SoccerNet-v2 上熵最好**。预算起初每步 1%，数据量超过 15% 后每步 2%，超过 25% 后每步 5%，超过 40% 后每步 10%。自适应调度把约 100 步降到 30 步。片段内的帧级分数用 max 汇聚比 mean 好。
- **报告结果**（NetVLAD++，SoccerNet-v2，loose Avg-mAP）：
  - 学习曲线下面积（AULC）：RS 45.11，UM 46.24，EM 47.96；
  - 只用 5% 数据时：RS 32.06，EM 37.24；
  - **达到全量性能的 90% 所需数据**：RS 45%，UM 31%，**EM 16%**；加上更快的调度后 EM 只需 13%，RS 需 40%；
  - SoccerNet-ball 上达到 90% 所需数据：EM 8%，RS 23%。头球数据集上 EM 12%，RS 35%。
  - 摘要原文："using only one-third of the dataset"。
- **接入 RallyIQ**：每批新上传的业余视频先跑现有管线，按回合切片段，计算片段熵（片段内每帧击球概率的二元熵取 max），再叠加几项 RallyIQ 特有的信号：DP 约束冲突（同侧连续击球、间隔 < 0.5 s、回合时长和击球数不匹配）；物理残差有峰值却没有击球候选；TrackNet 与 InpaintNet 结论不一致。按"场馆 × 机位"分层抽样，保证低机位侧拍被覆盖。预算调度照抄 1% → 2% → 5% → 10%。

### 6. TFD：面向少标注精确事件定位的时序特征蒸馏（Xu, Wei, Wells, Aryal；arXiv 2607.10998，2026-07）

- 论文：<https://arxiv.org/abs/2607.10998>
- 代码：<https://anonymous.4open.science/r/TFD-8535>（匿名评审仓库，论文 HTML 中给出）
- **许可**：论文没有声明许可，匿名仓库无法核实 → 按保留所有权利处理。
- **方法**：DINO 式 EMA 教师-学生。教师是学生的 EMA，基础衰减 0.9995，快速衰减 0.99；教师看弱增强片段，学生看强增强片段（时序运动增强：16×16 网格里按运动选 9 个 patch，概率 0.5；外加帧遮挡，长度 3，最多 5 段）。对齐损失作用在**主干的逐帧特征**上（ℓ2 归一化后算余弦距离，教师侧停梯度），而不是投影头输出。总损失 L = L_sup + λ(t)·L_align：前 5 个 epoch 只用监督损失，之后 10 个 epoch 内 λ 按余弦升到 1。主干是带 TGS 时序移位的 ViT-S，训练 50 epoch，片段长 100 帧。
- **报告结果**（mAP，δ = 1 帧）：
  - 10% 标签：Tennis **80.19**（ASTRM 76.61，E2E-Spot 64.38，T-DEED 36.69）；FSComp 38.29；FSPerf **50.29**（比 ASTRM 的 45.75 高 4.54）；FineDiving 52.87。
  - 80% 标签：Tennis 93.04，FSPerf 84.52。
- **接入 RallyIQ**：现在的击球检测是"候选 + GBM"，不是端到端视频模型，短期用不上。将来如果改用 E2E-Spot / T-DEED 一类模型做击球帧定位，可以把所有无标签业余视频当蒸馏数据，只标 10% 的片段，并照抄上面的 warm-up 和 λ 日程。

### 7. TENT 与测试时 BN 统计量重估（Wang, Shelhamer 等，ICLR 2021）；附 CoTTA（CVPR 2022）

- 论文：<https://openreview.net/forum?id=uXl3bZLkr3c>；CoTTA：<https://arxiv.org/abs/2203.13591>
- 代码：<https://github.com/DequanWang/tent>；<https://github.com/qinenergy/cotta>
- **许可**：TENT **MIT**（已读）；CoTTA **MIT**（已读）。
- **报告结果**（TENT README，CIFAR-10-C 严重度 5，WRN-28-10，平均错误率）：未适应时 43.5%；只重估 BN 统计量（"norm"）**20.4%**；再对 BN 仿射参数做熵最小化（TENT，每个 batch 更新一次）**18.6%**。其中运动模糊一项从 34.8% → 14.2% → 13.7%。CoTTA 在 TENT 的基础上加了"增强平均的伪标签 + EMA 教师 + 随机恢复部分权重"，针对的是长时间、持续变化的域，仓库里同时提供 AdaBN 和 TENT 的基线。
- **接入 RallyIQ**：已确认 TrackNetV3 的 `Conv2DBlock` 使用 `nn.BatchNorm2d`（读了 `model.py`）。最便宜的做法：每段新视频先把模型设为 train 模式，用前 200-500 帧前向一遍，只更新 BN 的 running stats、不反传，然后切回 eval 正式推理。在低机位测试视频上对比检出率即可验证。如果要做 TENT 式的梯度更新，热图没有分类熵，可以改用"检测点到阻力 ODE 拟合曲线的距离"作损失，只更新 BN 仿射参数，并加回滚保护（损失变大就恢复）。一个球馆连拍几小时的场景可以参考 CoTTA 的随机恢复，防止漂移。

### 8. CoTracker3：用伪标签在真实视频上训练点跟踪（Karaev 等，Meta，2024）

- 论文：<https://arxiv.org/abs/2410.11831>
- 代码：<https://github.com/facebookresearch/co-tracker>
- **许可**：LICENSE.md 为 **CC BY-NC 4.0**（已读）。README 说明部分组件另有许可：Particle Video Revisited 用 MIT，TAP-Vid 和 LocoTrack 用 Apache-2.0。主体代码和权重不可商用。
- **报告结果**：README 称其 "trained with 1000x less data than previous top-performing models"。先在 Kubric 合成数据上训练，再在真实视频上用伪标签微调。
- **配方**：多个教师（CoTracker2、CoTracker3 online/offline、TAPIR）组成集合，**每个 batch 随机选一个教师**生成伪轨迹；学生只在教师认为可见的点上算损失。
- **接入 RallyIQ**：不直接用这个模型（不可商用，而且穿梭球太小，不适合通用点跟踪）。借鉴的是流程："合成预训练 → 多教师随机伪标签 → 真实视频微调"。RallyIQ 的教师集合可以是：微调后的 TrackNetV3、原版 TrackNetV3、另一个检测器，以及阻力 ODE 插值。每个 batch 随机选一个教师，避免学生过拟合某一个教师的系统误差。

### 9. TTP：测试时个性化的人体姿态估计（Li, Hao, Di, Gundavarapu, Wang，NeurIPS 2021）

- 论文：<https://arxiv.org/abs/2107.02133>；项目页 <https://liyz15.github.io/TTP/>
- 代码：<https://github.com/harry11162/TTP>
- **许可**：**MIT**（已读）。
- **报告结果**（论文 Table 1，Transformer 关键点版本，mPCK / mAcc）：

  | 数据集 | 基线 | 联合训练（不做 TTP） | TTP 在线 | TTP 离线 |
  |---|---|---|---|---|
  | Human3.6M | 85.42 | 87.90 | 91.70 | 92.05 |
  | Penn Action | 85.23 | 86.16 | 87.75 | 88.98 |
  | BBC Pose | 88.69 | 90.19 | 92.51 | 92.21 |

  在线模式每来一帧更新一次。只更新共享编码器和自监督头，监督头和 Transformer 冻结，Adam，学习率 1e-4。
- **接入 RallyIQ**：RTMO 是单阶段模型，没有 TTP 需要的自监督关键点分支，直接接入成本高。可以借鉴的思路是：对同一球员的一段视频，用图像变换等变性（翻转、缩放后关键点应一致）作为测试时损失，只微调最后几层。这适合低机位下身被网遮挡、手臂快速挥拍的情况。优先级低于 TrackNet 的 BN 重估。

### 10. ViTTA：动作识别的视频测试时自适应（Lin 等，CVPR 2023）

- 论文：<https://arxiv.org/abs/2211.15393>
- 代码：<https://github.com/wlin-at/ViTTA>
- **许可**：**无 LICENSE 文件**（查过 LICENSE、LICENSE.md 和 LICENSE.txt，GitHub API 的 license 字段也是 null）→ 保留所有权利。
- **方法**：在线把测试视频的特征统计量对齐到训练统计量，同时约束同一视频不同时间增强视图的预测一致。
- **接入 RallyIQ**：如果球种分类从 GBM 换成视频或骨架网络，可以用"特征统计对齐 + 时序增强一致性"做场馆级自适应。当前的 GBM 用不上，列为备选。

### 11. ETH ANYmal 腿足机器人打羽毛球（Ma, Cramariuc, Farshidian, Hutter；Science Robotics 2025）：感知噪声模型

- 论文：<https://arxiv.org/abs/2505.22974>（DOI 10.1126/scirobotics.adu3922）
- 代码：没有找到公开仓库。
- **许可**：没有可用代码。
- **内容**：
  - 仿真用的阻力模型是 m·dv/dt = m·g − m‖v‖v/L，**实测气动长度 L = 4.1 m**（即 C ≈ 0.244 m⁻¹）。
  - **感知噪声模型**：让相机移动、观察一个固定位置的球，用动捕得到相机到球的距离和角速度，把"检测概率"和"位置误差"都回归成距离和角速度的线性函数，再在仿真里按同样的函数注入噪声，使仿真和实机的感知误差水平一致。
  - 相机 60 fps；从快门到算出球位置的延迟为 60-160 ms；EKF 过程噪声标准差 2e-3，测量噪声 4e-2。
- **接入 RallyIQ**：第 1 条配方的检测噪声目前是常数（8 px 抖动、5% 漏检）。更好的做法是照 ANYmal 的思路，**在真实低机位视频上统计 TrackNet 的漏检率和定位误差，拟合成"离相机距离 × 图像角速度"的函数**，再注入合成轨迹。这样远端半场的高速杀球会自动得到更高的漏检率，更贴近 RallyIQ 的真实失败模式。

### 12. 羽毛球空气动力学参数汇总（仿真参数来源）

| 来源 | 模型 | 数值 | 换算成 C = 1/L（m⁻¹） |
|---|---|---|---|
| Cohen, Darbois Texier, Quéré, Clanet, "The physics of badminton", New J. Phys. 17 (2015) 063001（开放获取 PDF：<https://core.ac.uk/download/47088083.pdf>） | M·dU/dt = M·g − ½ρSC_D‖U‖U，L = 2M/(ρSC_D) | 通用 L = 4.6 m，U∞ = √(gL) ≈ 6.7 m/s，C_D = 0.65 ± 0.05；羽毛球 M = 5.0 g，L = 4.04 m；尼龙球 M = 5.3 g，L = 4.48 m，C_D = 0.68 ± 0.05；有记录的最高球速 137 m/s | 0.217 / 0.248 / 0.223 |
| MonoTrack 代码（第 2 条） | d²x/dt² = g − C‖v‖v | C 初值 0.2，逐拍拟合范围 [0, 0.4] | 0.2 |
| ETH ANYmal（第 11 条） | 同上 | 实测 L = 4.1 m | 0.244 |
| Humanoid Whole-Body Badminton（arXiv 2511.11218；GitHub 上的 <https://github.com/Humanoid-Badminton/Humanoid-Whole-Body-Badminton-via-Multi-Stage-Reinforcement-Learning> 标题为 Multi-Stage RL，可能是同一工作的仓库，目前只有项目页，"All code will be released soon"，无 LICENSE） | 同上 | L = 3.4（论文未写单位）；初始条件 p_x ~ U(5, 8)，p_y ~ U(−2, 2)，p_z ~ U(−0.5, 2.5)，v_x ~ U(−25, −13)，v_y ~ U(−3, 3)，v_z ~ U(9, 18)；生成 200 万条，按可击球区过滤后保留 19,685 条 | 0.294 |
| Monocular Badminton Speed Estimation（第 13 条） | v̇ = g − k‖v‖v，k = g/v_T² | 采用文献值 v_T = 6.70 m/s；自由落体实测拟合 v_T = 6.97 ± 1.13 m/s | 0.219（实测 ≈ 0.20，1σ 范围约 0.15-0.29） |

- **翻转（turnover）**：Cohen 2015 给出击球后球体的振荡时间 τ_o 在 18.6 m/s 时约 102 ms，稳定时间 τ_s 在 20 m/s 时约 0.3 s。"Badminton shuttlecock stability: Modelling and simulating the angular response of the turnover"（Proc. IMechE Part P, 230(2):111-120, 2016，DOI 10.1177/1754337115596481）用欠阻尼二阶传递函数拟合翻转角响应，比较了 Li-Ning A+600 羽毛球、Yonex Mavis 350 和 Mizuno NS-5 尼龙球，结论是羽毛球阻尼比最高、时间常数最小、翻转最稳定（本次只读到摘要，没有具体数值）。
- **接入 RallyIQ**：仿真时每拍随机采样 C ∈ [0.18, 0.30] m⁻¹（覆盖羽毛球和尼龙球、磨损程度、场馆差异），中心值约 0.22。初速按球种分布，上限约 120 m/s。击球后 0.1-0.3 s 的翻转段用"短时额外阻力 + 方向滞后 + 更大的检测抖动"来近似（这是工程近似，不是论文结论）。它在 30fps 下对应击球后 3-9 帧，正好是转折点检测最容易出错的窗口。

### 13. Monocular Badminton Speed Estimation with Air Drag and Court Geometry（Huang 等，arXiv 2611.00031，2026-11）

- 论文：<https://arxiv.org/abs/2611.00031>
- 代码 / 数据：论文写 "The code used for the analysis will be made publicly available"，没有给链接；轨迹、标定和标注"可向作者索取"，原始视频不公开。
- **许可**：未声明。
- **报告结果**：用双手机立体（240fps）作参考，评估 51 个击球（吊球、平抽、半杀、杀球；76 个事件中有 51 个通过残差门限）。单目阻力拟合与带阻力立体参考之间的中位绝对速度差：自由拟合 13.3%，**加"过网平面"约束后 8.6%**，oracle 3.5%；只考虑重力、不加阻力时约 36.7%。拟合目标是图像残差的 soft-L1，尺度 a = 4 px；过网时刻的搜索范围为观测区间 ±50 ms。
- **接入 RallyIQ**：两点可以直接用。一是**阻力不可省略**：只考虑重力时速度误差约 37%，所以合成数据和物理残差过滤都必须带阻力。二是"球必须过网平面"这个约束可以加进 RallyIQ 的 ODE 拟合，用来判断两个击球候选之间的轨迹是否合理，同时把 soft-L1（a = 4 px）作为伪标签过滤的残差尺度。这篇用手机拍摄，与 RallyIQ 的设备一致。

### 14. SynthNet：用合成数据从单目视频估计 3D 轨迹（Ertner, Konglevoll, Ibh, Graßhof；ACM MMSports 2024）

- 论文：DOI 10.1145/3689061.3689073（ACM DL 开放获取，论文本身 CC BY 4.0）；ITU 页面 <https://pure.itu.dk/en/publications/synthnet-leveraging-synthetic-data-for-3d-trajectory-estimation-f/>
- 代码：没有找到公开仓库。
- **许可**：论文 PDF 是 CC BY 4.0，没有代码。
- **内容**：网球，端到端管线。先用 GRU 检测击球和弹地，把视频切成若干段；再用全连接网络从 2D 轨迹估计每段的初始条件，**只用物理仿真的合成数据训练**。损失直接算在估计的初始条件上，避开了"欧拉积分 + 相机投影"反传时的训练不稳定。摘要没有给具体数值（ACM PDF 本次没能抓取）。
- **接入 RallyIQ**：和第 1 条思路相同，但用的是轻量 MLP，适合部署在端侧。"先分段、再回归初始条件"的结构可以直接对应 RallyIQ 的"击球候选 → 逐拍 ODE 拟合"。另外，"在初始条件空间算损失"的技巧可以避免可微积分器带来的不稳定。

### 15. 其他可选条目（简述）

- **From Skeletons to Pixels：通过表示和预测蒸馏做少样本精确事件定位**（Yeoh, Kan；arXiv 2604.22839，2026-04；本科毕业论文）：骨架模态到像素模态的蒸馏，在 F3Set-Tennis(sub) 和花样滑冰上做 k-clip 少样本实验。摘要没有给数值，也没有代码链接。可作为"RTMO 骨架 → 视频模型"蒸馏的参考。
- **Self-supervised Feature Extraction for Enhanced Ball Detection on Soccer Robots**（Lin 等，arXiv 2506.16821，2025-06）：先用通用预训练模型生成伪标签，再做上色、边缘检测和 triplet 三种前置任务，加 MAML 快速适应新场地。数据集为 RoboCup SPL 的 1 万张标注图。摘要没有给数值，也没有代码链接。
- **LATTE-MV**（arXiv 2503.20936；代码 <https://github.com/sastry-group/LATTE-MV>，**MIT**，已读 LICENSE）：从单目乒乓球视频做规模化 3D 重建，再用重建结果训练对手动作预测 Transformer。仿真中机器人回球率从 49.9% 提到 59.0%。代码可商用，但数据集托管在 Hugging Face，许可未核实。
- **ALPES：精确事件定位的主动学习**（<https://github.com/zackjh/alpes>）：在 F3Set / AdaSpot 上做主动学习实验的代码。README 没有结果，**无 LICENSE 文件**，AdaSpot 部分的安装说明还没写完。可以关注。
- **PPAL：即插即用的目标检测主动学习**（CVPR 2024，<https://github.com/ChenhongyiYang/PPAL>，**Apache-2.0**，已读 LICENSE）：先按不确定性、再按多样性的两段式采样。对穿梭球这种单点热图不完全适用，可参考其采样结构。
- **FiftyOne**（<https://github.com/voxel51/fiftyone>，Apache-2.0）和 **Lightly**（<https://github.com/lightly-ai/lightly>，MIT）：做嵌入去重和"多样性 + 不确定性"选帧的工程工具，可以承载配方 D 的标注队列。

---

## 四、给 RallyIQ 的落地配方

### 配方 A：羽毛球物理合成轨迹 → 低机位随机相机 → 击球和落点模型（从零真实标签起步）

1. **仿真**：状态 (x, v)，dv/dt = g − C‖v‖v，步长 dt = 1/(fps × 30)，与 MonoTrack 一致。每拍从击球点生成到下一击球点：
   - 击球点按站位分布采样（前场、中场、后场），初速方向按球种采样（高远、杀球、吊球、平抽、挑球、网前搓推）；
   - C ~ U[0.18, 0.30]；初速大致为：杀球 40-110 m/s，高远 25-45 m/s，网前 3-10 m/s；
   - 球进入下一击球者可达范围（离地 0.3-3 m，身体周围 1.5 m 内）时截断，得到击球时刻标签；没被接到的球给出落点和界内外标签；
   - 用 Humanoid 论文的做法"大量生成再按可达区过滤"，保证分布合理。
2. **相机**：模拟业余机位。底线后 1-6 m 或侧边 1-5 m，高度 0.4-1.6 m，俯仰 −5° 到 25°，水平视场角 60-120°，DJI 广角加径向畸变 k1 ∈ [−0.3, 0]，帧率从 {25, 30, 50, 60} 中取，并像 Kienzle 那样按真实时间戳编码。多场地球馆再加"邻场轨迹"作为误检源。
3. **噪声**：从 Kienzle 的常数版（8 px 抖动、5% 误检、5% 漏检、运动模糊偏移）起步，再按 ANYmal 的思路换成在真实低机位视频上拟合的"漏检率和抖动 = f(到相机距离, 图像角速度)"。遮挡用 TT4D 的经验分布：统计真实视频里连续可见段和连续遮挡段的长度，交替采样。击球后 3-9 帧加大抖动，模拟翻转。
4. **模型**：输入 2D 轨迹 + 球场 4 角 + 网柱顶的像素坐标，用带时间戳 RoPE 的小 Transformer 或 GRU，输出每帧击球概率和击球方。先纯合成训练；再用少量真实低机位标注微调，或者只把它的输出概率作为一个新特征接进现有 GBM。
5. **验证**：用已标注的低机位测试视频（58 个击球），在 ±2 帧容差下看召回能从 34/58 提到多少。同时在 ShuttleSet 上确认精度没有下降。

### 配方 B：稀疏标注 + 物理插值伪标签（减少 TrackNet 的标注量）

- 每段视频只标每回合的击球帧，以及每拍中间 2-3 个点；其余帧用每拍独立的阻力 ODE 拟合插值（可以加过网平面约束，见第 13 条）。
- soft-L1 残差（a = 4 px）低于 3 px 的帧作为伪标签；残差大的帧在损失里屏蔽（SST 的忽略区）。
- 用这些伪标签微调 TrackNetV3，再用新模型重跑，迭代 1-2 轮。SST 的经验是第二轮还能再涨约 1.4 mAP（10% 标签：43.7 → 45.1）。

### 配方 C：击球检测自训练闭环（照 TT4D 的过滤器写）

1. 用现有管线跑完所有无标签业余视频。
2. 伪击球只有同时满足以下条件才采纳：GBM 分数 ≥ τh；符合 DP 约束（两侧交替，间隔 ≥ 0.5 s）；该点前后两段 ODE 拟合的重投影残差都低；球过网平面；腕关节在击球帧附近靠近球。
3. 回合级过滤：球可见率 ≥ 50%，至少 2 次击球。记录每个过滤器的拒绝数，跟 TT4D Table 1 一样逐阶段统计。
4. GBM 分数在 [τl, τh) 区间的候选不进训练（忽略）。
5. 在 ShuttleSet + 伪标签上重训 GBM，最后用少量真实低机位标注校准阈值。
6. 每轮把冲突最大的片段送去主动学习标注（配方 D）。

### 配方 D：主动学习选片段

score = w1·max_t H(p_hit(t)) + w2·(DP 约束冲突数) + w3·(物理残差有峰值但没有击球候选) + w4·(TrackNet 与 InpaintNet 不一致) + w5·1/(该场馆或机位已有的标注量)。

以回合为单位，每轮取 top-k 送标。预算按 Giancola 的调度：起初每轮 1%，超过 15% 后每轮 2%，超过 25% 后每轮 5%，超过 40% 后每轮 10%。目标是参照 Giancola 的结果（熵采样用 16% 数据达到全量性能的 90%），用约 1/5 到 1/3 的标注量达到全量效果。可以用 FiftyOne 或 Lightly 做标注队列和嵌入去重。

### 配方 E：测试时自适应

每段视频推理前，把 TrackNetV3 设为 train 模式，前向前 200-500 帧，只更新 BN running stats，然后切回 eval 推理。可选：用"物理一致性"作损失更新 BN 仿射参数，每段视频结束后重置；长时间连拍参考 CoTTA 的随机恢复。RTMO 不建议在线更新（多人场景容易漂移）。

---

## 五、许可风险汇总

| 条目 | 代码 | 权重 / 数据 | 能否商用 |
|---|---|---|---|
| Kienzle CVPRW25 / WACV26 | GPL-3.0 | mediastore 托管，未声明许可 | 只借鉴配方，自己重写 |
| MonoTrack | Adobe Research License（非商业） | TrackNetV2 数据集 | 不能用代码，可借鉴思路 |
| TT4D | CC BY-NC-SA 4.0（抬升模块 GPL-3.0） | 57 GB 数据集未单独声明；源视频来自 YouTube | 不可商用，只借鉴设计 |
| SST | BSD-3-Clause | 权重未声明；SoccerNet-v3 需签协议 | 代码可用，权重别用 |
| Giancola AL | 没有公开代码 | SoccerNet-v2 | 只借鉴方法 |
| TFD | 匿名仓库，未声明 | 公开基准 | 只借鉴方法 |
| TENT / CoTTA | MIT / MIT | — | 可用 |
| CoTracker3 | CC BY-NC 4.0（部分组件 MIT / Apache-2.0） | 同主仓库 | 不可商用 |
| TTP | MIT | Penn Action 等 | 代码可用 |
| ViTTA | 无 LICENSE 文件 | — | 保留所有权利 |
| ETH ANYmal / Humanoid badminton | 没有代码（Humanoid 仓库只有项目页，无 LICENSE） | — | 只借鉴参数 |
| Badminton Speed Estimation | 尚未发布 | 可向作者索取，未声明许可 | 只借鉴方法 |
| SynthNet | 没有代码 | 论文 CC BY 4.0 | 只借鉴方法 |
| LATTE-MV | MIT | Hugging Face 数据集，许可未核实 | 代码可用 |
| PPAL | Apache-2.0 | — | 可用 |
| FiftyOne / Lightly | Apache-2.0 / MIT | — | 可用 |
| ALPES | 无 LICENSE 文件 | — | 保留所有权利 |

---

## 六、局限

- 本次在容器中用 curl 只能访问 GitHub raw，huggingface.co 和 export.arxiv.org 都被拦截。论文数字来自 arXiv 的 abs、HTML 或 PDF 页面和 CVF PDF（经 WebFetch 摘要），没有逐页人工校对。SynthNet 的 ACM PDF 抓取失败，所以没有具体数字。2016 年翻转论文只读到摘要。
- Kienzle 和 TT4D 的数据与权重许可没有在托管页面上找到声明，按保留所有权利处理。
- 本地下载与代码检查目录：`/tmp/claude-0/-home-user/b8ae9930-87df-5663-af9e-aa30a1da687c/scratchpad/r2/dl/ll/`（Kienzle、MonoTrack 克隆，以及各仓库的 LICENSE 文件）和 `/tmp/claude-0/-home-user/b8ae9930-87df-5663-af9e-aa30a1da687c/scratchpad/r2/dl/ll3/`（TT4D、LATTE-MV、TrackNetV3 `model.py`）。
