# RallyIQ 调研：击球分类 / 球场标定 / 战术与 LLM 教练 / 业余数据集（2023–2026 新进展）

> 注：本文件是原始调研稿。许可证和数字已另行逐条核查，更正见上级目录 `2026-11-28-web-research.md` 第三节和 `verification.json`；两者不一致时以核查结果为准。


调研日期：2026-11-28。已排除任务中列出的已知项（ShuttleSet/22、CoachAI-Projects、BadmintonDB、MM25-FineBadminton、BST、TemPose、Shuttlecock Trajectory Dataset、AICUP 2023、jayclim/BadmintonAI、Badminton-AI-coach、badminton-pipeline-repro）。

**许可证核实方法**：代码许可证取自 GitHub API（`license.spdx_id`）或仓库 LICENSE / README 原文；权重和数据许可证取自 HuggingFace、Zenodo 或 Roboflow 页面。"未声明"表示我查看的页面没有写许可证。仓库没有 LICENSE 文件时，默认保留所有权利：可以阅读和复现思路，但未经授权不能把代码并入产品。论文页的 arXiv 许可证只覆盖论文本身，不覆盖数据和代码。

**两个总体结论**：
1. 目前**没有**大规模、许可宽松、低机位业余视角的羽毛球击球分类数据集。最接近的是 VideoBadminton（自拍、底线后方高机位、校队练习赛），但它的许可证未声明。业余视角的标注仍要靠自己积累：在约 150 条可信标签的基础上做主动学习。
2. 多个 2025–2026 基准表明，通用 VLM 直接看原始视频理解回合的能力很差（准确率约 30–46%，人类约 89%）。RallyIQ 现有的做法应保持不变：先用 CV 输出结构化数据，再让 LLM 写文字。下面第 17–20 条给出了具体的增强模式。

---

## 一、最值得做的 5 件事（按投入产出排序）

| # | 行动 | 依据 | 预期收益 / 成本 | 许可证风险 |
|---|---|---|---|---|
| 1 | **给 court_v2 GBM 加上"前后拍上下文"特征，用两阶段推理**：第一阶段照常预测；第二阶段把上一拍和下一拍的预测类型（one-hot 或概率）加上 ShuttleSet 统计出的类型转移先验作为新特征，再预测一次 | TemPose-TF-ASF（2026）在 ShuttleSet 31 类上把准确率从 0.835 提到 0.854，Macro-F1 从 0.742 提到 0.761，并且对 BlockGCN、SkateFormer、BST 都有效；ViSTec 和 F3ED 的上下文模块思路相同 | 几天的工作量，不需要新标注；在 77% 已到平台期的情况下，这是最便宜的增益 | 只借用思路，无风险（论文未放代码） |
| 2 | **用 CourtKeyNet 作为可学习的四角点候选器，并用它的置信度决定是否回退到手动标注**：把 CourtKeyNet 的四角点和现有直线检测的结果交叉验证，两者不一致或置信度低时提示用户手动标四角 | CourtKeyNet（MLWA 2026）专门检测羽毛球场四角，代码和权重均为 MIT。它自带三种置信度信号：热图峰值、热图熵、四边形几何合法性 | 直接针对"侧面角度自动检测出错"的问题；可以用自有画面加 Roboflow CC BY 4.0 数据微调，作者称微调需要 5k–7k 张图 | 低（MIT；数据由 CC BY 4.0 数据衍生，需署名） |
| 3 | **用点加线联合优化来精修单应性/相机参数，同时估计镜头畸变**：用 DeepLSD/ScaleLSD 提取亚像素级线段，再参考 TVCalib 的"线段重投影误差"可微优化（含畸变）做精修 | PnLCalib 和 NBJW 证明点加线优化明显优于只用点；TVCalib 是 MIT 许可的实现，并支持镜头畸变（DJI 广角相机必须处理畸变） | 能修正侧面角度和广角畸变导致的整体偏移；需要 1–2 周把足球模板换成羽毛球场模板 | PnLCalib 和 NBJW 是 GPL-2.0，不能并入闭源产品，只借鉴思路；TVCalib、DeepLSD、ScaleLSD 都是 MIT |
| 4 | **增加 RGB 视频分支**：在击球帧前后约 1 秒的片段上，用冻结的 V-JEPA 2 或 VideoMAE V2 小模型提特征，接一个注意力探针（attentive probe）。先在 VideoBadminton 上预训练探针，再用自有约 150 条标签微调，最后和坐标特征融合 | 坐标特征无法区分同一位置出手的吊、杀、高远等动作。VideoBadminton 上 SlowFast 达到 82.8% Top-1；V-JEPA 2 ViT-L 在细粒度的 Diving-48 上探针准确率为 89.0% | 中等成本，需要 GPU 推理；预期在易混类别上收益最大 | V-JEPA 2 代码为 MIT；VideoMAE V2 代码为 MIT，但权重需填表申请；VideoBadminton 的数据许可证未声明，仅限研究，商用前需征得作者同意 |
| 5 | **改造 LLM 报告的生成管线，让结论有据可查**：(a) 把统计数据以 CSV 加思维链的形式喂给 LLM（BADGE 的结论）；(b) 用短期记忆（最近 K 个回合）加长期记忆（累计球员统计）组织上下文（TennisExpert）；(c) 加一个"核查代理"，逐句对照结构化回合日志检查教练文字，删除无依据的说法（MADR） | TennisExpert 的 CIDEr 为 43.71，零样本的 Gemini-3-Pro 只有 10.11；MADR 去掉辩论和核查后，Hit@1 从 55.65% 降到 40%；通用 VLM 在 TennisTV 和 DoublesEval 上只有 30–46% | 主要是提示词和管线改动，成本低，能直接降低教练文字里的幻觉 | 只借用思路；BADGE 代码为 MIT |

**附加第 6 项（双打专用）**：双打中"谁击的球"可以这样判断：对每名球员的姿态序列做嵌入，打出"是否击球"的分数，取分数最高者，低于阈值则记为无击球（见第 7 条）。这样可以直接复用单打模型，代价很小。

---

## 二、条目详情

### （1）击球分类方法与数据集

**1. TemPose-TF-ASF：前后拍上下文融合（2026）**
- 论文：<https://arxiv.org/abs/2605.02558>（清华大学（新竹），Liu & Lee）
- 代码：未找到公开仓库，GitHub 搜索无结果。许可证不适用。
- 指标：ShuttleSet 清洗后共 40 场比赛、33,429 拍、31 类。准确率 / Macro-F1：TemPose-TF-ASF 为 0.854 / 0.761，TemPose-TF 为 0.835 / 0.742，BST-CG-AP 为 0.838 / 0.723。加上 ASF 后，BST-CG-AP 的准确率从 0.838 提到 0.843。
- 接入 RallyIQ：ASF 本质上就是"相邻拍类型嵌入后经 MLP 融入分类器"，第二阶段使用第一阶段的预测，因此推理时不需要真实标签。在 GBM 里只需加入 `prev_type_proba` 和 `next_type_proba` 两组特征，第二遍训练时这些特征要用 out-of-fold 预测生成，以防泄漏。这是 Top 1 行动项。

**2. VideoBadminton（2024）**
- 论文：<https://arxiv.org/abs/2403.12385>；数据和代码：<https://github.com/qilimk/VideoBadminton>，数据放在 Google Drive。
- 许可证：仓库没有 LICENSE 文件，数据也未声明许可证。arXiv 论文为 CC BY-NC-SA 4.0，但这只覆盖论文本身。
- 内容：18 类（按 BWF 标准），7,822 个片段，共 145 分钟。由研究者自行拍摄：60fps，1280×960，摄像机放在底线后约 2m、高 4.5m、俯角 30°，拍摄对象是中央大学校队的 19 名球员。
- 指标（Top-1 / 类平均）：SlowFast 82.80% / 73.80%；Swin3D 81.99%；PoseC3D 80.76%；R(2+1)D 79.53%；ST-GCN 74.41%。
- 接入 RallyIQ：这是目前最接近"非转播、底线后方机位"的公开击球数据集，可用于 RGB 分支预训练和类别映射，与 Top 4 配合。注意它的机位比三脚架低机位高很多，存在领域差。商用前需要联系作者确认许可。

**3. 视频基础模型：V-JEPA 2 / 2.1 与 VideoMAE V2**
- V-JEPA 2：<https://github.com/facebookresearch/vjepa2>。代码为 MIT，README 称"项目大部分为 MIT"，少数数据增强文件为 Apache-2.0；HF 模型卡上的权重许可证我没能核实。提供 ViT-L/H/g（2.1 版另有 ViT-B 80M、ViT-G 2B），并附 SSv2、Diving48 和 EK100 的注意力探针配置。Diving-48 准确率：ViT-L 为 89.0%，ViT-g/384 为 90.2%。
- VideoMAE V2（CVPR 2023）：<https://github.com/OpenGVLab/VideoMAEv2>。代码为 MIT；**权重需要填写 Google 表单申请**，权重条款未核实。提供蒸馏的 ViT-S 和 ViT-B，其中 ViT-S 在 K400 上为 83.7%，已集成进 MMAction2。
- 接入 RallyIQ：在击球帧前后取 16 帧，用冻结的骨干网络提特征，只训练探针。这样只需少量标签，适合 RallyIQ 约 150 条可信标签的情况。V-JEPA 2.1 ViT-B 和 VideoMAE V2 ViT-S 的大小适合离线批处理。
- 我没有找到在羽毛球上微调这些模型的公开工作。InternVideo2 只找到零散的个人仓库，没有纳入。

**4. ProtoGCN（CVPR 2025 Highlight）：骨架细粒度识别与原型学习**
- 论文：<https://arxiv.org/abs/2411.18941>；代码：<https://github.com/firework8/ProtoGCN>（**MIT**），权重在 Google Drive 和 HF。
- 指标：README 称在 NTU60/120、Kinetics-Skeleton 和 FineGYM 上达到 SOTA，具体数值我没有逐项核实。
- 接入 RallyIQ：如果保留 BST 骨架分支，可以用 ProtoGCN 替换或对比。它把动作拆解成可学习的"运动原型"，专门用于区分轨迹相似的动作，例如正手高远球和正手杀球。另外，原型和嵌入最近邻天然适合小样本场景（自有约 150 条标签）。同方向还有小样本骨架识别模型 PAINet（ICCV 2023，Apache-2.0，<https://github.com/starrycos/PAINet>）。

**5. F3Set / F3ED（ICLR 2025）与 ViSTec（AAAI 2024）：多标签事件与上下文修正**
- F3Set：<https://arxiv.org/abs/2504.08222>，代码 <https://github.com/F3Set/F3Set>。**仓库没有 LICENSE 文件**；视频只提供 YouTube 链接，不再分发。
- F3Set 的羽毛球子集（`data/f3set-badminton`）只有 58 个训练片段、932 个事件，来自转播精华（巴黎奥运），视频需向作者索取。它的标签按元素组合：`near/far × left/middle/right × forehand/backhand × 类型 × 方向(cc/dl/dm) × 结果(in/winner/err)`。
- 指标：在网球高粒度设置下，F3ED 的 F1evt 为 40.3，F1elm 为 75.2，Edit 为 74.0。
- ViSTec：<https://arxiv.org/abs/2402.15952>，乒乓球项目。用技术转移图作为上下文先验，帧级准确率 83.5%，Edit 76.3。论文没有给出代码，项目页为 <https://ViSTec2024.github.io/>。
- 接入 RallyIQ：(a) 把 RallyIQ 的标签从"单一类别"改成"多属性"（类型、正反手、方向、结果），GBM 可以对每个属性分别建模，组合起来更稳，也让 LLM 报告更细。(b) 用双向 GRU 或转移图对整个回合的预测序列做合法性修正，例如发球只能出现在回合第一拍，这和 Top 1 是同一类做法。

**6. RacketVision（AAAI 2026 Oral）：球拍姿态加球的跟踪**
- 论文：<https://arxiv.org/abs/2511.17045>；代码：<https://github.com/OrcustD/RacketVision>（**MIT**）；数据：<https://huggingface.co/datasets/linfeng302/RacketVision>（HF 页面标为 **MIT**）；模型：`linfeng302/RacketVision-Models`。
- 内容：羽毛球、乒乓球、网球的转播片段共 1,672 个（435k 帧）。球拍标注为框加 5 个关键点，共 24,621 个；球点标注 64,042 个。
- 结论：多项目联合训练能提升球的检测（网球 TrackNetV3 从 68.7 提到 81.9 mAP）。**直接拼接球拍特征反而有害**，改用交叉注意力融合才有提升。
- 接入 RallyIQ：球拍 5 个关键点（拍头方向、拍面朝向）是区分正手和反手、杀球和吊球的强线索。可以用它的 RacketPose 模型给每一拍加上"击球帧拍面角"特征。按论文结论，不要直接拼接，应使用注意力融合或手工几何特征。

**7. 用单打模型分析双打（2025）**
- 论文：<https://arxiv.org/abs/2508.13507>；代码：<https://github.com/100-heon/badminton_double_analysis>（**没有 LICENSE 文件**），README 称含 ST-GCN 和 Transformer 的预训练权重。
- 方法：YOLOv11x 加 BoT-SORT 跟踪，跟丢时用速度外推找回原 ID；ViTPose 提取 17 个关键点后以髋部为中心归一化；用 NT-Xent 对比学习训练 ST-GCN 嵌入，前场和后场各用一个骨干；对每名球员打"是否击球"分数，取最高者，低于阈值则判为无击球。
- 指标：双打"是否击球"整体准确率 0.6686（男双 0.7020，女双 0.6389），纯视觉基线为 0.5941，召回率约 81%。
- 接入 RallyIQ：RallyIQ 支持双打，但击球分类前要先确定"谁打的这一拍"。可以照搬这个思路：每名球员一个击球分数，取最大值并设阈值。另外，"前场 / 后场分开建模"也值得在 GBM 中作为分组特征使用。

**8. BadminSense（CHI 2026）：业余球员、手机视频加智能手表**
- 论文：<https://arxiv.org/abs/2603.21825>；数据：<https://github.com/taizhouchen/BadminSense_Dataset>（Google Drive，14GB）。
- 许可证：**CC BY-NC-ND 4.0**，即禁止商用、禁止演绎。
- 内容：12 名业余俱乐部球员（5 名初级、5 名中级、2 名高级），共 848 拍，4 类：正手高远、杀球、吊球、反手高远。每拍有 IMU、音频、手机视频片段，以及 21 名资深球员给出的 1–5 质量评分和拍面击球点。
- 指标（跨用户）：击球分类 91.43%；质量评分 MAE 0.438（1–5 分制）；击球点归一化误差 12.9%。
- 接入 RallyIQ：这是少有的业余球员加质量评分数据，可以用来**离线评估** RallyIQ 的"动作质量 / 教练文字"是否和专家评分一致。由于 NC-ND，不能用来训练商用模型。

**9. MultiSenseBadminton（Scientific Data 2024）**
- 论文：<https://www.nature.com/articles/s41597-024-03144-z>；数据：<https://doi.org/10.6084/m9.figshare.c.6725706.v1>；代码：<https://github.com/dailyminiii/MultiSenseBadminton>（**MIT**）。
- 数据许可证：文章本身为 CC BY 4.0，但 Data Records 和 Usage Notes 部分没有写数据许可证；figshare API 被代理拦截，我没能核实。
- 内容：25 人（12 名新手、8 名中级、5 名专家），共 7,763 次挥拍，只有正手高远和反手平抽两类，发球机供球。数据包含眼动、17 个 IMU、EMG、足底压力，以及正面、侧面、全景三路视频，另有教练给出的 1–7 分。
- 接入 RallyIQ：类别太少，对分类帮助有限。它的价值在于**按技术水平分级的视频**（新手 vs 专家），可用于训练"技术水平 / 动作规范度"打分器，作为 LLM 教练文字的依据。

**10. 新的转播数据集：BFMD、Fine-Badminton（Zenodo）、Shot2Tactic-Caption**
- BFMD（2026）：<https://arxiv.org/abs/2603.25533>。19 场 BWF 转播完整比赛（12 场单打、**7 场双打**），1,687 个回合，16,751 拍，12 种击球类型，并有击球描述文字（GPT-4.1 生成后人工核对）。论文**没有给出数据链接和许可证**。基于 VideoMAE 的描述生成模型在 singles 子集上 CIDEr 为 36.8。
- Fine-Badminton（2026-05）：<https://zenodo.org/records/20292976>。Zenodo 元数据为 **CC BY 4.0**，但描述中写着"仅供非商业学术研究"，两者矛盾，按更严的理解处理。内容为 10 场 YouTube 转播单打，29 类，9,817 个帧级时序标注，6.5GB。
- Shot2Tactic-Caption（MMSports 2025）：<https://arxiv.org/abs/2510.14617>。10 场男双转播，5,494 个击球片段，544 个战术单元，含 9 种战术类型和 5 种战术状态（开始、继续、中断、恢复、结束）。没有代码和数据链接。
- 接入 RallyIQ：三者都是转播视角，对低机位分类帮助有限。可借鉴之处：BFMD 的双打击球类型和描述文风可以作为 LLM 报告的模板；Fine-Badminton 可用于预训练击球时刻定位（hit spotting）；Shot2Tactic 的战术状态体系可以作为双打报告的战术词表。

### （2）球场检测与单图相机标定

**11. CourtKeyNet（Machine Learning with Applications 2026）**
- 论文：<https://www.sciencedirect.com/science/article/pii/S2666827026000496>；代码：<https://github.com/adithyanraj03/CourtKeyNet>（**MIT**）；权重：<https://huggingface.co/Cracked-ANJ/CourtKeyNet>（HF 页面写明 **MIT**，可商用）；数据：<https://github.com/adithyanraj03/Paper_09_Data-Set_CourtKeyNet>（README 写 MIT，但数据合并自 Roboflow 上两个 **CC BY 4.0** 数据集，因此需要署名）。
- 方法：Octave 特征提取，加极坐标变换注意力，加热图与回归混合的关键点定位，加四边形约束模块和几何一致性损失。输出 **4 个外角点**。README 提供三种置信度：热图峰值、熵、几何合法性（凸性、面积、长宽比）。
- 指标（作者在自己的验证集上测得）：微调模型 PCK 99.99%、IoU 99.67%；预训练模型 PCK 99.09%、IoU 94.33%。预训练用了 14 万张以上羽毛球和网球场图片，微调用了 7 千张以上。
- 接入 RallyIQ：这是 Top 2 行动项。它直接对应 RallyIQ 的"手动四角回退"：先用 CourtKeyNet 给出四角初值，用来约束或验证直线检测结果；置信度低时再请用户手动标注。需要用自有侧面角度和低机位画面验证，因为作者的指标来自偏转播和俯视的图片。

**12. TVCalib（WACV 2023）**
- 论文：<https://arxiv.org/abs/2207.11709>；代码：<https://github.com/MM4SPA/tvcalib>（**MIT**）。
- 方法：先对场线做语义分割，再以"线段重投影误差"为目标，可微地优化相机内参、外参和**镜头畸变**，并带自验证阈值。用于足球。
- 接入 RallyIQ：这是许可宽松的"线段精修"参考实现（Top 3）。做法是把足球场模型换成羽毛球场线模型（含双打边线、单打边线、前后发球线、中线），输入 RallyIQ 已检测到的线段像素。DJI 广角画面的径向畸变正是导致单应性在边角偏差大的原因，TVCalib 的畸变项可以直接解决这个问题。

**13. PnLCalib 与 No Bells, Just Whistles（NBJW，CVPRW 2024）**
- PnLCalib：<https://github.com/mguti97/PnLCalib>。代码为 **GPL-2.0-or-later**；发布的权重为 **CC BY 4.0**，可商用但需署名；权重在足球场上训练。
- NBJW：<https://arxiv.org/abs/2404.08401>，代码 <https://github.com/mguti97/No-Bells-Just-Whistles>（**GPL-2.0**）。2023 年 SoccerNet 标定挑战的方案，利用几何性质（直线交点、椭圆切点）扩充关键点。
- 方法：用 HRNet 热图检测关键点和线段端点，再用线段交点补充关键点数量，接着做初始标定，最后用 **PnL（点加线）非线性精修**。
- 接入 RallyIQ：借用两个思路，(a) 用检测到的线段两两求交来补充关键点，(b) 在 PnL 精修中联合使用点和线。**GPL 代码不能并入闭源的 RallyIQ**，应使用 TVCalib（MIT）或自行实现。

**14. 深度线段检测器：DeepLSD、ScaleLSD、HAWP**
- DeepLSD（CVPR 2023）：<https://github.com/cvg/DeepLSD>，**MIT**。用深度图像梯度检测线段，并做亚像素精修。
- ScaleLSD（CVPR 2025）：<https://github.com/ant-research/scalelsd>，**MIT**。可扩展的深度线段检测。
- HAWP（TPAMI 2023）：<https://github.com/cherubicXN/hawp>，**MIT**。线框解析，能直接输出线段和交点。
- 接入 RallyIQ：替换 Hough 或 OpenCV LSD 作为球场线提取前端。在业余画面里球场线常常是低对比、反光、地胶拼接或运动模糊，深度检测器的召回率和端点精度更好，也能给 Top 3 的精修提供更好的输入。注意（我的推断，未验证）：线框类模型会把地板拼缝等干扰也当成线，需要先做颜色或宽度过滤，再匹配球场模板。

**15. 参考实现：Good-Badminton 与 RallyOS**
- Good-Badminton：<https://github.com/yo-WASSUP/Good-Badminton>，**Apache-2.0**，约 1.2k stars。2026-06 新增"按标准羽毛球场线模型匹配白 / 黄线的自动边界检测"，并可在 Gradio WebUI 中手动修正四角；姿态部分支持 RTMPose、RTMO 和 YOLO-Pose。
- RallyOS：<https://github.com/nakuldagade/RallyOS>，**没有 LICENSE 文件**，只能借鉴思路。用 top-hat 形态学加 LSD 提线，再用 RANSAC 匹配多种球场模板，能自动判别项目，例如区分羽毛球场（13.40×6.10m）和匹克球场；用 Zhang 单平面法自标定相机；保留 4 点点击回退。
- 接入 RallyIQ：Good-Badminton 可以作为对照基线，在 RallyIQ 的侧面角度样本上直接比较自动检测的成功率；它是 Apache-2.0，代码可以参考。RallyOS 的"内部发球线参与模板评分"和"单平面自标定"值得借鉴，但不能复制代码。

**16. 球场关键点数据与模型：Roboflow Universe 与 Hit-frame Keypoint-RCNN**
- Roboflow 上的羽毛球场数据集（页面均标 **CC BY 4.0**）：
  - BadmintonC：<https://universe.roboflow.com/himani-jain/badmintonc>，3,647 张，检测框格式，是 CourtKeyNet 的数据来源之一。
  - badminton court key point 1/2/3：<https://universe.roboflow.com/tennis-fyp/badminton-court-key-point-2>。
  - Badminton-calibration：<https://universe.roboflow.com/gerdos-flow/badminton-calibration>。
  - Badminton Court Keypoints：<https://universe.roboflow.com/badminton-analysis-w9ude/badminton-court-keypoints>，图片数和许可证页面上没有显示。
  - 除 BadmintonC 外，各数据集的图片数和机位页面上都没有写，需要下载后检查。
- Automated Hit-frame Detection（2023）：<https://arxiv.org/abs/2307.16000>，代码 <https://github.com/arthur900530/Automated-Hit-frame-Detection-for-Badminton-Match-Analysis>（**MIT**）。包含**球场 Keypoint-RCNN 权重**和击球帧检测（Transformer），数据在 Google Drive，未声明许可证。
- 接入 RallyIQ：Roboflow 的 CC BY 4.0 数据加上自有侧面角度帧（手动四角标注会自然积累这些帧），可以用来微调 CourtKeyNet，或训练一个多点 YOLO-pose 球场关键点模型，例如 RallyIQ 现有模板的 20 多个交点。Keypoint-RCNN 可以作为另一个候选器，做集成投票。

### （3）战术分析、LLM/VLM 教练

**17. TennisExpert（2026）：结构化解析加分层记忆加 VLM 解说**
- 论文：<https://arxiv.org/abs/2603.13397>（新加坡国立大学、南洋理工、Griffith）；代码：<https://github.com/LZYAndy/TennisExpert>（**没有 LICENSE 文件**），数据只提供 YouTube URL 和标注，限学术用途。
- 内容：TennisVL 包含 202 场大满贯转播，40,523 个回合片段。管线为：解析比分牌、击球和落地事件（Edit 81.2）、球员和球的球场坐标，然后组织短期记忆（最近 K=4 个回合）和长期记忆（累计统计），最后用 SFT 微调 Qwen3-VL-8B。
- 指标：CIDEr 43.71，LLM 评分 88.05。零样本最佳的 Gemini-3-Pro 分别为 10.11 和 59.89。
- 接入 RallyIQ：报告生成可以照搬"短期加长期记忆"的上下文组织方式。每个回合的教练点评只附最近 4 个回合的结构化事件，再加上全场累计统计，不要整场一次性塞给 LLM（Top 5）。

**18. MADR：用多代理辩论做战术回合检索（CVPR 2026 Findings，由 CVF 补充材料链接推断）**
- 论文：<https://people.cs.nycu.edu.tw/~yushuen/data/BadmintonRetrieval26.pdf>（阳明交通大学，Zhang & Wang）；项目页：yixiang1120.github.io/MADR-project-page，没有看到代码仓库和许可证。
- 方法：把 ShuttleSet 的回合转成**结构化文本比赛日志**（拍序列加 UniPose 姿态描述）。进攻分析员和防守分析员两个代理辩论，总结员撰写叙述，两名审阅员给反馈，**核查代理逐句对照日志检查**；最后对叙述做向量检索和重排序。
- 指标：Hit@1 55.65%、Hit@10 83.04%。VLM/CLIP 基线 Hit@1 只有 5.65%；去掉辩论后降到 40%。
- 接入 RallyIQ：(a) 增加核查步骤，降低教练文字的幻觉（Top 5）；(b) 支持"找出我所有挑后场后丢分的回合"这类自然语言检索，作为报告的交互功能。

**19. 用 LLM 写报告和教练反馈：BADGE（IJCAI 2024 Workshop）与可解释动作质量评估（ERA 2026）**
- BADGE：<https://arxiv.org/abs/2406.18116>，代码 <https://github.com/AndyChiangSH/BADGE>（**MIT**）。结论：GPT-4 在**CSV 输入加思维链**时报告质量最好；同时用 LLM 做评分员评估报告，GPT-4 的评分与人工相比偏好 GPT-4 生成的报告，存在自偏好偏差。
- 可解释动作质量评估：<https://www.aimspress.com/article/doi/10.3934/era.2026333>（Electronic Research Archive 34(10)，2026，文章为 **CC BY 4.0**，没有代码）。管线为 MediaPipe 关键点，加 mST-GCN++ 分类（11 类，94.34%，自有数据），加 6 个指标（挥拍速度、手腕轨迹平滑度、腕和肘姿态、步法效率、控拍），再经分层模糊推理得到质量等级和雷达图，最后由 LLM 生成反馈（专家评分 8.594/10）。
- 接入 RallyIQ：RallyIQ 的报告输入应统一为 CSV 或表格，并要求 LLM 先列证据再下结论。可以先算出 6 类可解释指标，再交给 LLM 措辞，避免 LLM 自己"看出"动作问题。用 LLM 评估报告时要换一个不同的模型做评委，以避开自偏好偏差。

**20. VLM 体育理解基准：证明"直接让 VLM 看视频"仍不可靠**
- TennisTV（2025）：<https://arxiv.org/abs/2509.15602>，基于 F3Set 构建，9 类任务，2,943 个问题。最佳 GPT-4.1 为 46.2%，人类 89.0%。帧采样方面，单拍级任务一直到 64 帧都在提升，回合级任务在 32 帧时最好。
- DoublesEval（2026）：<https://arxiv.org/abs/2608.24439>，专业羽毛球双打，约 9.6k 个问题，4 个层级。零样本 VLM 只有 27.6–32.3%；不需要训练的 TacticCheck（用模型自己的低层预测重排高层答案）能提升约 7–8 个点。论文提到 GitHub 仓库，但正文没有给出链接，也没有许可证。
- SPORTU（ICLR 2025）：<https://arxiv.org/abs/2410.08474>，仓库 <https://github.com/chili-lab/SPORTU>（**没有 LICENSE 文件**）。视频部分覆盖 7 项运动（含羽毛球），12,048 个问答对。选择题最佳为 Qwen2-VL-72B 的 70.94%；困难的规则类问题只有约 52–58%。"先推理后作答"反而更差。
- SportR（2025–26，<https://arxiv.org/abs/2511.06499>，<https://github.com/chili-lab/SportR>）和 DeepSport（2025，<https://arxiv.org/abs/2511.12908>，基于 Qwen2.5-VL-7B，覆盖 12 项运动含羽毛球，总分 40.08，GPT-5 为 35.70；没有代码和许可证）。
- TennisVAR（2026）：<https://arxiv.org/abs/2608.12920>，仓库 <https://github.com/WhynotGit2025/TennisVAR>（**没有 LICENSE 文件**）。采用"事件、关系、证据、战术"范式：先检测击球事件，再用图推理选出证据拍，最后由 Qwen3-VL-8B 加 LoRA 生成答案。战术体系为 6/17/25 三级，分层 F1 为 70.98。
- 接入 RallyIQ：这些基准是一条设计约束：VLM 只能在**已有结构化证据**的基础上做解释，不能直接作为击球或战术标签的来源。如果确实要用 VLM，可以借鉴 TacticCheck 的自洽性重排，以及 TennisVAR 先选证据拍再作答的做法，并参考 TennisVAR 的三级战术体系设计 RallyIQ 的战术标签。

**21. CourtSI / CourtSI-Bench（2026）：球场空间智能**
- 论文：<https://arxiv.org/abs/2603.09896>；代码：<https://github.com/Visionary-Laboratory/CourtSI>（**Apache-2.0**）；数据：HF `Charlie019/CourtSI-1M`（HF 页面为 **apache-2.0**，9.52GB）；模型：HF `Charlie019/CourtSI-Qwen3-VL-8B`，模型许可证未核实。
- 内容：基于 RacketVision 的羽毛球、网球和乒乓球画面，用球场几何做 PnP 标定，半自动生成 100 万以上的空间问答对（计数、距离、定位、关系）。微调后的 Qwen3-VL-8B 在 CourtSI-Bench 上从 37.7% 提到 61.2%；Gemini-3-Pro 为 64.6%，人类为 73.6%。
- 接入 RallyIQ：如果报告里要回答"这一拍我离网多远"这类空间问题，应该优先用 RallyIQ 自己的单应性直接计算。CourtSI 的价值在于：其"用球场几何做 PnP 生成度量真值"的数据引擎和 RallyIQ 的标定思路相同，可以借来批量生成训练或评测样本；也可以当作 VLM 问答功能的空间能力基线。

### （4）可用于训练或评测的其他数据（含业余视角）

**22. 其他数据集一览**
- **Mendeley 羽毛球动作视频（2026-06）**：<https://data.mendeley.com/datasets/3sp4xntp34/1>，**CC BY 4.0**。884 段视频，一部分是在室内球场用手机或相机多角度拍摄，另一部分来自 YouTube；按帧组织，页面上**没有看到击球类别标注**。用途：作为业余画面的无标注素材，用于测试球场检测的鲁棒性和自监督预训练。注意：YouTube 部分的版权许可证可能覆盖不到。
- **无人机双打数据集（MTAP 2023）**：<https://github.com/Ning-D/Drone_BD_ControlArea>（**没有 LICENSE 文件**）。顶视和后视无人机视频，带 4 名球员和球的框，用于双打"控制区"估计。用途：可以借鉴双打控制区和覆盖热图这类战术指标。
- **RacketDB（VISAPP 2025）**：<https://github.com/muhabdulhaq/racketdb>，数据在 HF `muhabdulhaq/racketdb`（**CC BY 4.0**，247GB）。22,682 张图、16,045 个球拍框，源自上面的无人机视频。用途：训练球拍检测器，作为 RacketVision 球拍姿态之外的补充。
- **Extended OpenTTGames（2025）**：<https://arxiv.org/abs/2512.19327>，<https://github.com/moamal01/table_tennis_data>，标注为 **CC BY-NC-SA 4.0**。乒乓球侧面固定机位，120fps，1,457 拍，并标注了身体前后倾和双脚状态。用途：标签设计可以借鉴，为 RallyIQ 增加"重心或步法状态"属性，作为教练文字的依据。不能商用。

---

## 三、许可证速查

| 资源 | 代码 | 权重 | 数据 | 能否用于商用产品 |
|---|---|---|---|---|
| CourtKeyNet | MIT | MIT（HF） | MIT（衍生自 CC BY 4.0，需署名） | 可以 |
| TVCalib / DeepLSD / ScaleLSD / HAWP | MIT | — | — | 可以 |
| PnLCalib / NBJW | GPL-2.0(+) | CC BY 4.0（PnLCalib） | — | 代码不能并入闭源产品；权重可以用（需署名） |
| Good-Badminton | Apache-2.0 | 见仓库 Release | — | 可以 |
| RacketVision | MIT | HF 有模型，未逐一核实 | MIT（HF） | 可以；原始画面为转播，版权风险自行评估 |
| CourtSI | Apache-2.0 | 未核实 | Apache-2.0（HF） | 基本可以 |
| V-JEPA 2 | MIT | README 称整体 MIT，HF 卡片未核实 | — | 大概率可以，需复核 |
| VideoMAE V2 | MIT | 需填表申请，条款未核实 | — | 需复核 |
| ProtoGCN / BADGE / Hit-frame Detection | MIT | 见仓库 | Hit-frame 数据未声明 | 代码可以 |
| PAINet | Apache-2.0 | — | — | 可以 |
| VideoBadminton | 没有 LICENSE | — | 未声明 | 需联系作者 |
| F3Set / TennisExpert / SPORTU / TennisVAR / 双打迁移 / RallyOS | 没有 LICENSE | — | 多为 YouTube 链接 | 只能借鉴思路 |
| BadminSense | — | — | CC BY-NC-ND 4.0 | 不能（可用于内部评测研究） |
| Fine-Badminton（Zenodo） | — | — | CC BY 4.0，但描述写"仅非商业学术" | 按非商业处理 |
| Extended OpenTTGames | — | — | CC BY-NC-SA 4.0 | 不能 |
| Roboflow 球场数据集 / RacketDB / Mendeley 884 段视频 | — | — | CC BY 4.0 | 可以（需署名） |
| MultiSenseBadminton | MIT | — | 数据许可证未核实（文章为 CC BY 4.0） | 需复核 |
| TemPose-TF-ASF / ViSTec / BFMD / MADR / Shot2Tactic / DoublesEval / ERA 2026 | 未公开 | — | 未公开或未声明 | 只能借鉴思路 |

## 四、检查过但未纳入或证据不足的项

- **THETIS、TenniSet、TTStroke-21、P2ANet**：早于 2023 年，或者是转播或实验室视角，本次没有核实许可证，因此不列入推荐。P2ANet 仓库（<https://github.com/Fred1991/P2ANET>）没有 LICENSE 文件。
- **TennisCourtDetector**（<https://github.com/yastrebksv/TennisCourtDetector>，287 stars）：没有 LICENSE 文件，2024 年后不再维护。它的"热图检测 14 点、局部直线求交精修、再用单应性重建偏移点"后处理思路仍值得借鉴。
- **"A Deep Learning-Based Framework for Racket Sports Court Registration"**（Springer 2024，DOI 10.1007/978-3-031-63219-8_2）：HAL 全文被反爬拦截，无法核实内容，没有纳入。
- **BadmintonGPT**：GitHub 上只有 `cedricliang1129/BadmintonGPT`（0 stars，没有 LICENSE，基于 LangGraph 的多代理系统），没有论文或评测，参考价值低。
- **Kaggle**：只找到 BWF 排名和赛果表格类数据，以及一个 IMU 数据集"Motion Data for Badminton Shots"，没有可用的业余视频击球标注集。
- **调研限制**：WebFetch 的配额在中途用尽，所以部分 HF 模型卡（V-JEPA 2 权重、CourtSI 模型）和 figshare 的许可证没能核实，表中已标为"未核实"。
