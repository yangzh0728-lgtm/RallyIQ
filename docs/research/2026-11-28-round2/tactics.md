# RallyIQ 调研：球拍运动战术分析与"如何战胜这名对手"报告（2024–2026）

> 注：原始调研稿。许可证和数字已另行逐条核查，更正见 `../2026-11-28-web-research-round2.md` 第三节和 `verification.json`；不一致时以核查为准。


调研日期：2026-11-28。

**范围**：2024–2026 年羽毛球及其他球拍运动的战术分析，具体包括：
- 逐拍价值（类 xPoint / Shot Quality）
- 回合胜负预测
- 球员风格嵌入
- ShuttleNet / DyMF / RallyNet 之后的下一拍预测
- 对手建模
- 商业球拍运动产品向用户展示哪些指标

另外专门核查了 arXiv 2606.15992。

**已排除**：任务列表里注明"已调研"的条目（TrackNet 系列、WASB、ShuttleSet、CoachAI 主仓库、BST、TemPose、FineBadminton、TennisExpert、BADGE 等）。RallyNet 只作为基线出现，下文介绍的是它 2024–2026 年的后续工作。

**许可证判定方法**：
- 代码：读仓库根目录的 LICENSE 文件，并用 GitHub 搜索 API 交叉核对 `license.spdx_id`。
- 数据：读数据集页面。
- 没有 LICENSE 文件按"保留所有权利"处理，只能读、只能借鉴思路，不能拷代码。
- 没能亲自读到许可证的地方标"未核实"。

**数字来源说明**：论文数字大多来自 arXiv HTML 正文或 PDF，经网页抓取工具提取。个别数字可能有转录误差，落地前请回原文核对。

---

## 0. 结论速览

1. **先做"逐拍结果模型"，不要做"回合胜负预测"。** 同名开源项目 sarah-agrawal/rallyiq 用 ShuttleSet 做了一个现成的反例：
   - 从每一拍预测"这一回合谁赢"，准确率只有 52%；离回合结束还有 6 拍以上时只有 50%。
   - 改成预测"这一拍是制胜 / 失误 / 被回过来"之后，在未见过的比赛上，制胜分 AUC 达到 0.841，失误 AUC 0.653。
   - 所以 RallyIQ 的"击球价值"应该定义为**逐拍的 P(制胜)、P(失误) 及其变化量**，再按球员、场区、拍型汇总成"这名对手在哪里最容易丢分"。
2. **这类模型只需要"每分谁赢"这一个标签**，不需要逐拍专家标注：
   - PolyU 的离线 RL 论文（EAAI 2026，MIT 代码）用 Bradley-Terry 偏好模型，只从"回合赢家 vs 输家"学出逐拍奖励。
   - ICAART 2026 用 ΔQ 作为单拍价值。
   - 业余视频里"每分谁赢"可以由回合层自动推出（见 rally.md 的发球锚定层），推不出时让用户点一下头像补标（"多练"App 就是这么做的）。
3. **对手建模先做"条件分布 + 收缩"，深度序列模型放第二步。**
   - 一场业余比赛只有几百拍。RallyTemPose、MT-HTA 这类模型在 ShuttleSet 上 Top-1 只有 54%，macro-F1 只有 0.53，用在单个业余对手身上必然过拟合。
   - 先用分层贝叶斯 / 混合效应模型，向人群先验收缩，估计 P(拍型, 方向 | 击球区, 来球)。这里借鉴网球 SQS 论文的"发球方 × 接发方交叉随机效应"，把对手强度从指标里剥离。
   - 数据积累后，再接入 RallyTemPose 这类 MIT 许可、输入为"姿态 + 场地坐标"的模型。
4. **报告的组织方式参考 TactiPlay（HKUST，2026）。** 它面向业余羽毛球，结构如下：
   - 三个问题域：技术执行 / 站位与回位 / 战术选择，共 17 个子描述符，由国家级运动员标注得出。
   - 四个报告主题：失分模式 / 得分模式 / 战术意识问题 / 技术执行问题。
   - 每条结论按出现频次排序，并链接到对应的视频片段。

   在 16 人的用户研究中，引发的反思条目数是"只给统计"基线的近 2 倍（79 vs 44）。TactiPlay 承认没有逐条验证 LLM 诊断是否正确。RallyIQ 已有的落地校验（grounding check）正好补上这一点，可以作为差异化卖点。
5. **不要把战术推理交给视频 VLM。** DoublesEval（2026）测了 Qwen3-VL、VideoLLaMA3 等模型的职业双打战术推理：零样本准确率只有 27–32%，加上 TacticCheck 重排后仍低于 40%。LLM 只应该消费结构化事件数据。
6. **商业产品现状**：
   - 网球职业端（ATP/TennisViz）已有 0–10 的 Shot Quality、Conversion/Steal、进攻占比。
   - 匹克球的 PB Vision、PointIQ 已经做到"AI 教练找出体现你强弱项的片段"和"针对某一对对手做双打 scouting"。
   - 业余羽毛球产品（流行羽"翎析AI"、多练、手表类 App、Smashspeed）停留在：回合剪辑、TOP 球速、得失分热区、按回合长度统计得分率、挥拍计数。
   - **业余羽毛球目前没有产品提供"逐拍价值 + 对手弱点"报告**，这是 RallyIQ 的空位。
7. **arXiv 2606.15992**（网球）：
   - **只用姿态就检测击球时刻。** 方法是加权的手腕/肘/肩速度峰值，另有一个 BiGRU 变体。在 ±100 ms 容差下 F1 = 85.65%，平均时间误差 80.6 ms。
   - 代码**未公开**：v2 写的是"发表后公开"；v1 称"在 Kaggle 上"，但没给链接。所以没有代码许可证可言。论文文本为 CC BY 4.0。
   - 对 RallyIQ：适合作为击球打分器的**附加特征**（RTMO 手腕速度峰值），不能替代轨迹转折点。80 ms 在 30 fps 下约 2.4 帧，而它的方向和姿态质量分类接近随机水平。
8. **注意名称冲突**：
   - GitHub 上有一个同名羽毛球项目 sarah-agrawal/rallyiq（2026-10）。
   - 匹克球分析产品 PointIQ 部署在 `rallyiq.pages.dev` 域名下。

   如果 RallyIQ 要商用，建议尽早做商标检索。

### Top 5（按对"如何战胜这名对手"报告的价值排序）

| # | 名称 | 类型 | 许可证（已核实） | 为什么排这里 |
|---|---|---|---|---|
| 1 | **sarah-agrawal/rallyiq**：职业单打逐拍结果模型（制胜 / 失误 / 继续） | 类 xPoint，LR / XGBoost / 因果 Transformer | 代码：**无 LICENSE，保留所有权利**；数据：ShuttleSet / ShuttleSet22，仓库根 MIT | 最接近 RallyIQ 现有 GBM 技术栈：只用击球时刻已知的特征，按比赛切分评测，SHAP 可解释，还给出了"回合胜负不可预测"的反面证据。不能拷代码，但方法一天内能用自己的代码重写 |
| 2 | **Offline RL for Badminton Tactical Decision-Making**（PolyU，EAAI 2026） | 逐拍奖励（Bradley-Terry）+ 混合动作 CQL 策略 | 代码 **MIT** | 只需要回合胜负就能学出逐拍"功劳"，可以直接迁移到只有"每分谁赢"标签的业余数据。CQL 策略能给出"这个局面该打什么"的建议。商用友好 |
| 3 | **TactiPlay**（HKUST，arXiv 2607.27125） | 业余羽毛球复盘系统与问题分类法 | 无代码、无数据发布 | 唯一针对业余羽毛球、有用户研究验证的战术报告设计，可以直接作为 RallyIQ LLM 报告的信息架构模板 |
| 4 | **RallyTemPose**（ITU Copenhagen，CVPRW 2024） | 下一拍拍型预测（姿态 + 场地坐标 + 球员嵌入） | 代码 **MIT** | 输入正好是 RallyIQ 已有的 RTMO 姿态和单应性坐标，不需要球的落点。可作为"对手倾向模型"的第二阶段 |
| 5 | **CoachAI+ / RallyNetv2**（NYCU，SSAC 论文 20251430） | 球员行为模仿、对战模拟、战术分类、得失分状态统计 | CoachAI+ 仓库**无 LICENSE，保留所有权利**；前作 RallyNet 在 CoachAI-Projects 内，**MIT** | 给出了"如何战胜"报告的几个现成模块：三类战术（全场压迫 / 防守反击 / 四角调动）对应的胜率、Top-4 得分与失分状态、逐拍动量图。模拟的胜率与真实胜率误差（MAE）为 7.4% |

如需最多 8 条，后续依次是：
- **#6 DShuttleNet**：双打数据集，含协同指标 CCEI / TLAI 和 6 类失分原因。
- **#7 BadmintonVision**：MIT 代码，指标定义可以直接实现，包括路径效率、回位时间、中场占位、拍型熵。
- **#8 MT-HTA**：层级多任务下一拍预测，跨三项运动；代码保留所有权利。

---

## 1. 专项核查：arXiv 2606.15992《Player Disjoint Multi-Task Tennis Stroke Analysis from Monocular Pose Sequences》

- **链接**：
  - 论文：https://arxiv.org/abs/2606.15992
  - v1 题为《Multi-Task Tennis Stroke Biomechanics Analysis Using MediaPipe Pose》，2026-06-16。
  - v2 为 2026-11-06。
  - 作者：Jigyashman Hazarika、Prince Kumar、Karthik Vasu。
- **是否只用姿态检测击球时刻：是。**
  - 输入：MediaPipe Pose Landmarker 的 33 个关键点中保留 13 个。
  - "动力学击球事件检测器"：对持拍侧手腕速度、肘位移、肩位移加权，权重分别为 0.5 / 0.3 / 0.2（遵循动力链原理），取超过阈值的局部速度峰值作为击球时刻锚点。论文没有给出阈值和峰值显著度参数。
  - 复合变体：再结合运动学轨迹拐点和加速度过零点。
  - BiGRU 变体：2 层双向 GRU，从姿态轨迹逐帧输出击球概率。
  - 不使用球轨迹，也不使用音频。论文明说数据集里"没有球轨迹跟踪"。
- **报告结果**：
  - 击球检测：在 6 个未剪辑验证片段、206 个共识击球事件上，±100 ms 容差下 F1 = 85.65%（比基线阈值法高 3.24 个百分点），平均绝对时间误差 80.6 ms。
  - 拍型分类：准确率 74.27%，平衡准确率 65.37%，Macro-F1 0.645。
  - 方向分类：平衡准确率 37.75%（随机水平 33.3%）。
  - 姿态质量：平衡准确率 50.56%。后两项作者自己承认接近随机水平。
  - 评测协议为留一球员（7 名球员，3 个随机种子，嵌套交叉验证），共 1,281 个击球序列。
- **数据和机位**：
  - 机位"大多在底线后方"，偶有侧前方 45° 角度。
  - 转播与消费级相机混合，30–60 fps 统一重采样到 30 fps。
  - v1 称职业素材来自 YouTube 公开训练视频，业余素材由一名业余球员自拍。
- **代码 / 数据 / 许可证**：
  - v2 第 10 节写道，源码、配置和评测脚本"发表后将在公开仓库发布"；原始视频因涉及可识别个人信息**不能共享**；匿名化特征"在许可条款允许时可能发布"。
  - v1 的数据可用性声明写"代码和数据在 Kaggle 上"，但**没有给出链接**。
  - 目前**没有可用代码，也就没有代码许可证**。论文文本在 arXiv HTML 页标注为 **CC BY 4.0**（v1）。
- **对 RallyIQ 的意义**：
  - 这个检测器在网球上 ±100 ms 的 F1 为 0.86。RallyIQ 的问题是在低机位业余视频中漏检，58 拍只找到 34 拍。
  - 建议把"持拍手腕速度峰值"（来自 RTMO，按人框高度归一化）作为**一个额外特征**加入现有的梯度提升击球打分器，用来召回球在球员身前被遮挡、轨迹转折点缺失的击球。
  - 不建议单独使用，原因有三：
    1. 羽毛球的挥拍主要靠前臂内旋和手腕爆发，在 30 fps 下峰值经常只占 1–2 帧。
    2. 假动作、空挥，以及双打中同伴的挥拍，都会触发误报。
    3. 80 ms 的时间误差太粗，不足以做落点和球速估计。
  - 方法很简单，按论文描述自己实现即可，不存在许可风险。
  - 旁证：Baek & Yun 2025（下文条目 17）在双打上用纯姿态做"击球 / 非击球"分类，F1 只有 0.71，同样说明纯姿态只能做辅助。

---

## 2. 条目详解

每个条目包含：链接、许可证、报告精度与数据集，以及**接入 RallyIQ 的方式**。

### 1. sarah-agrawal/rallyiq：职业单打逐拍结果模型 ★Top 1

- **代码**：https://github.com/sarah-agrawal/rallyiq （2026-10，0 星）
- **许可证**：
  - **没有 LICENSE 文件**：GitHub API 返回 `license=null`；main 分支下的 LICENSE、LICENSE.md、LICENSE.txt 均为 404。按**保留所有权利**处理。
  - README 称数据来自 ShuttleSet / ShuttleSet22（CoachAI，MIT）。
- **任务定义**：
  - 每一拍打上三类标签之一：`winner`（制胜）、`error`（击球方这拍丢分）、`continues`（被回过来）。
  - 特征只用击球时刻已知的信息：双方站位、两人距离、击球高度、是否反手或头顶、距上一拍的时间、上一拍拍型、本拍拍型、比分（用上一回合重建，避免泄漏当前分）。
  - **刻意不用落点**，因为落点发生在决策之后。
- **数据与结果**：
  - 数据：ShuttleSet + ShuttleSet22 合并去重，共 94 场、7,622 回合、81,403 拍。按比赛切分：训练 75 场，测试 19 场（16,049 拍）。
  - 各模型在测试集上的表现：

    | 模型 | 制胜 AUC | 失误 AUC | 对数损失 |
    |---|---|---|---|
    | 基础率 | 0.50 | 0.50 | 0.377 |
    | 逻辑回归 | 0.832 | 0.642 | 0.334 |
    | XGBoost | 0.836 | 0.644 | 0.332 |
    | 2 层因果 Transformer | 0.838 | 0.642 | 0.334 |
    | **XGBoost + Transformer 集成** | **0.841** | **0.653** | **0.329** |

  - 约 91% 的拍是"被回过来"，所以准确率没有参考意义，作者用 AUC 评估。
  - **反面结论**：最初的版本是"从每一拍预测回合胜者"，准确率只有 52%；离回合结束还有 6 拍以上时只有 50%。
  - 附带一个"击球推荐器"：同一局面下把拍型逐一替换，用 XGBoost 打分，只保留职业选手在该区域实际使用率 ≥ 3% 的拍型。
- **接入 RallyIQ**：
  - 用自己的代码重写一个同构模型，**不要拷贝仓库代码**。
    - 特征全部换成 RallyIQ 能从业余视频算出的量：RTMO 脚点经单应性映射得到双方坐标；击球高度由 TrackNet 击球帧的球高相对肩、头关键点判断；拍型用现有 GBM 的**概率向量**，不用硬标签；节奏是击球间隔。
    - 先在 ShuttleSet 上预训练，再用业余视频"每分谁赢"的标签做保序回归（isotonic）校准。
  - 报告里输出：
    - 每名球员、每个场区、每种拍型的平均 P(制胜) 和 P(失误)。
    - "被迫失误 / 非受迫失误"的区分：失误发生前，对手上一拍的 P(制胜) 越高，越可能是被迫失误。
- **局限**：
  - 只用了职业数据。业余比赛的失误率高很多，必须重新校准。
  - 推荐器得出的是相关关系，不是因果关系。作者本人也这样声明。

### 2. Offline Reinforcement Learning for Badminton Tactical Decision-Making（PolyU，EAAI 2026）★Top 2

- **论文**：Liu, Tao, Huang，*Engineering Applications of Artificial Intelligence* 第 164 卷，113395，2025-11-30 在线发表。PDF：https://wenminggong.github.io/papers/Offline_RL_for_Badminton_EAAI_paper.pdf
- **代码**：https://github.com/Wenminggong/Offline_RL_for_Badminton ，**MIT**（LICENSE 原文为"Copyright (c) 2024 文名工"，GitHub API 也返回 MIT）。
- **方法**：
  - MDP 建模：
    - 状态：本方位置、对方位置、对方上一拍的拍型和落点。
    - 动作：拍型（10 类）、意图落点（连续二维）、击球后移动位置（连续二维）。
    - 奖励：只在终局给出，赢球 +1，输球 −1。
  - **Bradley-Terry 偏好奖励模型**：用"回合赢家轨迹优于输家轨迹"的成对比较，学出**逐拍奖励**，同时作为离线策略评估的替代。
  - **混合动作空间 CQL**：基于 SAC，对离散部分和连续部分分别设置熵权重。
- **数据与结果**：
  - 数据：ShuttleSet + ShuttleSet22 合并为 94 场、43 名球员，过滤后 5,416 个回合，按 7:2:1 切分。验证集和测试集包含训练时没见过的球员。
  - 奖励模型的回合偏好准确率：同分布 0.9692，分布外 0.9539。
  - 混合动作 CQL 的平均估计奖励：同分布 0.8703，行为策略 0.7254，最好的基线 BC 0.8482；分布外 0.8679，对应 0.7273 和 0.8581。
  - 论文**没有报告胜率**，也只评估了单步（短视）奖励。
- **接入 RallyIQ**：
  - **逐拍功劳分配**：Bradley-Terry 奖励模型只需要回合胜负，正好匹配业余数据的标签条件。它的状态特征（双方位置、来球拍型、来球落点）RallyIQ 都能从 TrackNet、RTMO 和单应性得到。可以用 ShuttleSet 上训练好的模型初始化，再用业余比赛微调，输出"每一拍的价值 r(s,a)"。
  - 报告可以直接写出"本场得分贡献最高 / 最低的 5 拍"，并附视频片段。
  - CQL 策略可以给出反事实建议，例如"在这个局面下，模型更偏好打对角网前"。但必须标注"基于职业数据的统计偏好"，并且只在样本覆盖充分的局面给出。

### 3. TactiPlay：面向业余羽毛球的多粒度战术解析与视频锚定复盘（HKUST，2026）★Top 3

- **论文**：https://arxiv.org/abs/2607.27125 （cs.HC，2026-07-29）。作者：Qiaoyi Chen、……、Xiaojuan Ma。
- **许可证**：页面未显示代码、数据或分类法的发布链接，也没有许可证声明。只能借鉴设计。
- **问题分类法**：
  - 3 名国家级运动员对 7 段业余比赛视频写了 304 条自由标注，开放编码后得到 **3 个问题域、17 个子描述符**：
    - 技术执行（Technical Execution）
    - 站位与回位（Positioning and Recovery）
    - 战术选择（Tactical Choices）
  - 单拍层面的问题归入技术执行；以 3 拍为一组的战术层面问题归入后两类。
- **系统流水线**（大部分自动，有人工复核）：
  - 回合切分：豆包 Doubao-seed-1.6 VLM。109 个回合上切分准确率 75%，赢家及赢球原因准确率 68%。
  - 球场：MonoTrack。人体：HRNet 姿态 + 单应性。羽毛球跟踪：WASB。
  - 击球检测：几何规则法，在 564 拍上 **P = 0.71，R = 0.72**，与 RallyIQ 的 34/58 处于同一水平，说明业余击球检测是行业共同瓶颈。
  - 拍型分类：用 VLM 看 11 帧窗口。论文说准确率"不足，需要人工修正"，每 8 分钟比赛约需 25 分钟人工。击球人识别准确率 77.9%。
  - 问题诊断：GPT-4.1 读入标注后的击球帧图像、事件特征和分类法描述，输出 JSON 格式的问题（类别、描述符、场地坐标证据），再生成回合摘要并聚类成子主题。
  - 作者承认**没有系统验证每条战术诊断是否客观正确**。
- **报告结构**：
  - 四个主题：失分模式 / 得分模式 / 战术意识问题 / 技术执行问题。子主题按频次排序，用星级显示，并链接到视频片段。
  - 胜负时间线（按颜色区分），以及每个回合的 Rally Explainer 文字摘要。
  - 可按回合过滤的击球方 / 接球方热力图。
  - 视频回放上同步叠加场地分区、骨架和球轨迹。
- **用户研究**：
  - 16 名业余球员（BadmintonCN 2–4 级），被试内设计。
  - 反思条目 79 vs 44。具体性、可执行性、恰当性均显著优于基线。
  - 但"对反馈的信任度"没有显著差异。
- **接入 RallyIQ**：
  - 直接采用"四主题 + 子主题频次排序 + 片段证据"的信息架构，以及三问题域分类法，作为 LLM 报告的输出 schema。
  - 与 TactiPlay 的区别：每条结论在落地校验里要求同时具备三样东西——事件 ID、计数和置信区间，从而补上它"诊断未验证"的缺口。
  - 其他可借鉴的设计：
    - 业余用户最常用的是"报告驱动"的浏览方式（12/16）。
    - 基线组的用户只会盯着技术执行（65.9%），对战术选择关注很少。这说明 RallyIQ 需要主动把战术结论推到前面。

### 4. RallyTemPose：用姿态和场地位置预测下一拍（CVPRW 2024）★Top 4

- **论文**：Ibh, Graßhof, Hansen（ITU Copenhagen），"A Stroke of Genius: Predicting the Next Move in Badminton"，CVPR 2024 Workshops，第 3376–3385 页。https://openaccess.thecvf.com/content/CVPR2024W/CVsports/papers/Ibh_A_Stroke_of_Genius_Predicting_the_Next_Move_in_Badminton_CVPRW_2024_paper.pdf
- **代码**：https://github.com/MagnusPetersenIbh/RallyTemPose ，**MIT**（LICENSE 原文为"Copyright (c) 2023 MagnusPetersenIbh"）。8 星。README 没有提到预训练权重。
- **方法**：
  - Transformer 编码器-解码器。
  - 输入：HRNet 2D 骨架、经单应性得到的双方脚点坐标、球员 ID 嵌入、历史拍型序列。拍型嵌入由预训练 BERT 对拍型文字描述编码得到。
  - **不使用羽毛球落点或轨迹。**
- **结果**（10 类拍型）：

  | 数据集 | Top-1 | Top-2 | Top-3 |
  |---|---|---|---|
  | ShuttleSet | 54.3% | 77.3% | 92.5% |
  | BadmintonDB | 62.8% | 83.5% | 93.1% |

  在 BadmintonDB 上，简单模型也表现不错，因为数据量小、大模型容易过拟合。
- **接入 RallyIQ**：
  - 输入和 RallyIQ 的输出完全对齐：RTMO COCO-17 可以替换 HRNet；单应性坐标现成。所以可以在 ShuttleSet 上训练，再用业余数据微调。
  - 产出"对手倾向"：给定场上局面，输出对手下一拍的拍型分布。报告里写成"对手被压到反手后场时，约 X% 回直线高远"这类结论，并给出它的**可预测性**（分布熵）。可预测性高，说明可以提前预判。
  - 新对手没有球员嵌入时，用"未知球员"token，或者只用本场数据做少量适配。
  - 第一阶段建议先用条件频率加收缩估计（见第 4 节）。

### 5. CoachAI+ Badminton Environment / RallyNetv2（NYCU，SSAC 论文）★Top 5

- **论文**：CoachAI+ Badminton Environment: Realistic Badminton Game Simulator for Enhancing Player Performance。MIT Sloan Sports Analytics Conference 研究论文，PDF 编号 20251430，网站显示为 SSAC 页面。
  - 页面：https://www.sloansportsconference.com/research-papers/coachai-badminton-environment-realistic-badminton-game-simulator-for-enhancing-player-performance
  - PDF：https://cdn.prod.website-files.com/5f1af76ed86d6771ad48324b/6791fce2ea6c69e3fbc7b5db_CoachAI%2B%20Badminton%20Environment.pdf
- **代码**：https://github.com/KuangDW/CoachAI-Plus （30 星）。**没有 LICENSE 文件**：API 返回 null，main 和 master 分支的 LICENSE* 均为 404。按**保留所有权利**处理。演示站：https://badminton.andyjjrt.cc/
- **前作 RallyNet**（ECML-PKDD 2024，https://arxiv.org/abs/2403.12406）：代码在 wywyWang/CoachAI-Projects 内，该仓库根目录 LICENSE 为 **MIT**。
  - 状态 17 维：比分、球位置、双方位置、对手移动向量等。
  - 动作：拍型、落点、移动位置。
  - 与真实胜率的平均差 0.0379，最大差 8.15%。
- **RallyNetv2 相对 RallyNet 的提升**：

  | 指标 | RallyNet | RallyNetv2 |
  |---|---|---|
  | 落点 DTW | 0.6449 | 0.2983 |
  | 拍型 CTC 损失 | 26.5384 | 8.3533 |
  | 移动 DTW | 0.3424 | 0.1934 |
  | 胜率 MAE | 9.5% | 7.4% |

- **功能模块**（README）：
  - **Shot Evaluation**：回合长度与体能消耗关系图、拍型可视化、逐拍"动量与节奏"图。
  - **Tactic Evaluation**：把回合归入"全场压迫 / 防守反击 / 四角调动"三类战术，计算各自的使用率与胜率。
  - **得失分原因统计**：Top-4 得分与失分的"状态 + 动作"组合，以及回合动画。
  - 环境内置基于 ShuttleSet 的动作约束：拍型约束、落点约束、移动距离约束。
- **接入 RallyIQ**：
  - "三类战术胜率"和"Top-4 得分与失分状态+动作"可以直接作为报告段落，用 RallyIQ 的结构化事件自己实现即可，逻辑很简单，不需要拷代码。
  - "模拟你 vs 对手"需要每名球员有大量数据，业余场景短期内做不到。可以在同一用户积累多场比赛后，作为"长期 scouting"功能。
  - RallyNet（MIT）可以作为研究基线。

### 6. DShuttleNet：羽毛球双打数据集与协同指标（2026）

- **论文**：https://arxiv.org/abs/2610.14016 （NYCU 等，Jhang、Chen、Lai……Huang）。
- **数据**：Kaggle DOI 10.34740/kaggle/dsv/16097938（https://www.kaggle.com/datasets/anonymousay58dki/dshuttlenet ）。**许可证未核实**：Kaggle 被代理拦截，抓到的页面也没有许可证字段。论文里没有许可证声明，也没有代码仓库。
- **内容**：
  - 2022–2025 年 BWF 世界巡回赛双打 58 场（男双 24、女双 24、混双 10），4,483 个回合，54,141 拍。
  - 每拍都有四名球员和羽毛球的 (x, y) 坐标；18 种拍型归并为 9 大类。
  - **6 类互斥的失分原因，以及主要责任人。**
  - 两个几何特征：队友相对角 TRA、垂直击球角 VSA。
  - 两个协同指标：**场地覆盖效能指数 CCEI**、**威胁负载不对称指数 TLAI**。
- **结果**：
  - TSCVAE：FDE 0.524，拍型准确率 0.504，F1 0.295。MoCVAE：FDE 0.972，准确率 0.466。
  - 去掉 VSA 后，拍型准确率只有 46–50%。
- **接入 RallyIQ**：
  - 双打报告可以照搬两个指标：用 CCEI 衡量对手组合的覆盖漏洞，用 TLAI 找出"承受威胁更多 / 更弱"的一方，即"打谁"。
  - 6 类失分原因加责任人，可以作为双打 LLM 报告的失分归因 schema。
  - 数据许可证核实之前，只用来设计指标，不用来训练商用模型。

### 7. BadmintonVision：精英羽毛球自动战术分析框架（BMC SSMR 2026）

- **论文**：Liang, Zhou, Zhang，*BMC Sports Science, Medicine and Rehabilitation* 18:434（2026-09-08）。https://pmc.ncbi.nlm.nih.gov/articles/PMC13595539/ 。文章许可为 CC BY-NC-ND 4.0。
- **代码**：https://github.com/liangyongjie51/BadmintonVision 。LICENSE 为 **MIT**（原文为"Copyright (c) 2025 The BadmintonVision Authors"），并附加声明：转播视频不属于仓库内容，版权归原权利人所有。GitHub API 因为这段附加声明显示为 NOASSERTION。论文称仓库含配置、标注指南和预训练权重。数据需向作者申请。
- **结果**：
  - 数据：2019–2025 年 5 届世锦赛单打转播，5,865 个回合片段，30,000 张专家标注图像，139,501 张无标注图像。
  - 检测：YOLOv12m 的 mAP@0.5 为 0.806；加 ConvNeXt V2 自监督预训练后为 0.885；再加 10 帧时序 MotionFormer 后为 **0.941**（按比赛切分）。
- **战术指标定义**（本条的主要价值）：
  - 平均移动速度
  - **路径效率**（直线位移除以实际移动距离）
  - 每回合变向次数
  - **中场占位率**
  - **回中（回位）时间**
  - **拍型香农熵**
  - 一阶马尔可夫转移矩阵及 lag-1 调整残差
  - 按回合阶段统计的拍型构成
  - 平均回合长度、跳杀终结率
- **接入 RallyIQ**：
  - 以上指标都能从 RTMO 脚点轨迹、单应性和拍型序列直接算出，可以作为"移动与站位"和"可预测性"两个报告段的指标。
  - 拍型熵低、转移矩阵集中，说明这名球员容易被预判。
  - 模型权重是在转播画面上训练的，RallyIQ 用不上，只借鉴指标定义。

### 8. MT-HTA / "One Hierarchy, Many Sports"：层级对齐的多任务下一拍预测（2026-11）

- **论文**：https://arxiv.org/abs/2611.15529 （Cai, Sun, Wu, Chen，2026-11-12）。
- **代码**：https://github.com/gndd1221/For-Sports-event-modeling-with-rally-and-score-hierarchical-A-Multi-Task-Hierarchical-Transformer 。**没有 LICENSE 文件**：API 返回 null，main 分支的 LICENSE* 均为 404。按**保留所有权利**处理。
- **方法**：
  - 按"拍 → 回合 → 局（网球多一层盘）"的层级建模。
  - 双路编码：层级时序 Transformer，加上把每个语义字段当作一个 token 的 iTransformer。
  - 每个任务有独立的注意力查询。
  - "轮次式风格门控"：把球员嵌入按"下一拍击球方 / 接球方"分别路由。
  - 羽毛球任务：拍型、是否反手、落点区域。
- **结果**（macro-F1，3 个随机种子）：

  | 数据集 | 多数类基线 | MT-HTA |
  |---|---|---|
  | 羽毛球 ShuttleSet | 0.2175 | **0.5309** |
  | 乒乓球（原版） | 0.2532 | 0.5315 |
  | 乒乓球（扩展） | — | 0.5525 |
  | 网球（Match Charting Project） | 0.5349 | 0.7095 |

  在乒乓球上，改造后的 ShuttleNet 只有 0.3958。
- **接入 RallyIQ**：
  - 可借鉴两点结构设计：把局 / 盘层面的上下文（比分压力、本局已打回合）作为对手模型的条件；用"击球方 / 接球方"角色区分球员嵌入。
  - 代码不能用。
  - 网球部分用到的 Match Charting Project 数据是 **CC BY-NC-SA 4.0（禁止商用）**，见第 5 节。

### 9. ICAART 2026：基于深度强化学习与博弈树的羽毛球最优击球预测

- **论文**：Minooka, Tahara, Ohsuga, Sei（电气通信大学），ICAART 2026 第 2 卷，第 1493–1504 页，DOI 10.5220/0014317600004052。https://www.scitepress.org/publishedPapers/2026/143176/pdf/index.html
- **许可证**：没有代码。论文页面没有许可证声明（未核实）。
- **方法**：
  - 用 LSTM 近似 Q 函数，SARSA 式更新。Q 值的含义是"从该状态和动作出发，最终得分的概率"。
  - **单拍价值 A(t+1) = Q(s_{t+1}, a_{t+1}) − Q(s_t, a_t)**，正值表示这一拍让局面更有利。
  - 回合按拍合并成博弈树，节点收益等于子拍价值之和，再用伪回合扩充。
- **结果**：
  - 数据：ShuttleSet 44 场过滤后的 2,220 个回合，预测 18 类（9 种拍型 × 2 个方向）。
  - Top-1 / 3 / 5：本文 0.346 / 0.669 / 0.815；ShuttleNet 0.315 / 0.548 / 0.780。
- **接入 RallyIQ**：
  - "ΔQ 作为单拍价值"与条目 1 的 ΔP(制胜) 是同一个思想，可以作为价值定义的备选。
  - 实现很简单，按论文公式自己写即可。

### 10. ShuttleEnv：数据驱动的羽毛球策略 RL 环境（北京大学，2026）

- **论文**：https://arxiv.org/abs/2603.17324 （Li, Gong, Chen, Lu, Ji, Wang, Yang, Li）。arXiv HTML 页标注 CC BY 4.0。
- **许可证**：没有代码仓库，也没有数据下载。只有一个演示视频。
- **方法**：
  - 不做物理仿真，用显式的概率模型描述回合动态，核心是两个模型：
    - **击球成功模型 M_succ**：这一拍会不会直接失误。
    - **回球模型 M_ret**：对手能不能接回。
  - 数据是作者手工标注的林丹 vs 李宗伟转播比赛（Lin-Lee Badminton Dataset）。
- **结果**：
  - 下一拍预测：拍型 Top-1 为 0.473（林）/ 0.438（李），Top-4 为 0.880 / 0.871；落点区域 Top-1 为 0.580 / 0.594。
  - PPO 智能体对固定对手的胜率 98.3%。
- **接入 RallyIQ**：
  - 把 xPoint 拆成两部分：P(这一拍没失误) × P(对手接不回)。这种分解对业余教练更直观，例如"你的杀球 85% 能打进，但 80% 被接回"。
  - 两个概率都能用业余数据的"制胜 / 失误 / 继续"标签直接拟合。建议和条目 1 一起实现。

### 11. "How good was my shot?"：乒乓球球员技能量化与风格嵌入（京都大学，2026）

- **论文**：https://arxiv.org/abs/2603.25736 （Kubota, Hasegawa, Kawahara, Nishino）。arXiv 非独占许可。
- **许可证**：没有代码。数据声明为"发表后发布处理后的数据"，**未发布**。
- **方法**：
  - HitFormer：从 2D 球轨迹估计击球向量。
  - HitFlow：Flow Matching 生成模型，以比赛上下文为条件（FiLM），联合学习每名球员的嵌入。
  - SkillNet：基于 RankNet，在嵌入上做技能排序。
- **结果**：
  - 数据：日本乒协 JTTA 70 场，53 名球员。
  - 嵌入可以解码出性别、持拍手、身高。
  - 与赛事排名的 Spearman 相关：男 0.665，女 0.699。两两排序准确率：男 77.3%，女 73.5%。
- **接入 RallyIQ**：给"风格嵌入"提供了一个路线：不直接对统计量做聚类，而是学习"每名球员在给定局面下的出球分布"的条件生成模型，把球员嵌入当作这个模型的参数。RallyIQ 积累到数百名业余用户后，可以支持两个功能：
  - "和你对手风格最像的球员"检索，复用对他们的已知对策。
  - 业余水平估计，作为对手强度的先验。

### 12. Court to Conversation：计算机视觉 + RAG-LLM 的羽毛球战术问答（KBS 2025/2026）

- **论文**：Bharadwaj & Srinivasa（PES University），*Knowledge-Based Systems* 333:115027，DOI 10.1016/j.knosys.2025.115027。https://www.sciencedirect.com/science/article/abs/pii/S0950705125020659
- **许可证**：没有代码，只在 Zenodo 上有两段演示视频：https://doi.org/10.5281/zenodo.16940656 。期刊文章付费。
- **方法**：
  - YOLO 检测，MediaPipe 姿态。
  - 击球检测用"羽毛球轨迹 + 姿态"多线索融合。
  - 把比赛结构化为战术知识库，再用 RAG-LLM 做自然语言问答。
  - 定义了一个 **"失位（out-of-position）"状态**，作为球员易受攻击的度量。
- **结果**：
  - 数据：5 场男单转播，约 4.5 小时，750 多个回合。
  - 检测 mAP：球员 0.94，羽毛球 0.83。全文被截断，没能看到 RAG 部分的评测。
- **接入 RallyIQ**：
  - "失位"指标可以这样实现：击球瞬间接球方离"理想回位点"的距离，并按对方击球区条件化。这可以作为"对手哪里容易被调动"的核心证据。
  - RAG 架构和 RallyIQ 现在"结构化事件 → LLM"的思路一致，佐证了这个方向。

### 13. DoublesEval：用职业羽毛球双打诊断 VLM 的多智能体战术推理（2026）

- **论文**：https://arxiv.org/abs/2608.24439 （Cheng 等，HKUST 与澳门大学）。
- **许可证**：论文说"代码在我们的 GitHub 仓库"，但**没有给出链接**。许可证未知。
- **内容**：
  - 4 场 BWF 超级 750 / 1000 双打，60 个精选回合，约 9,600 道题（单选题和简答题各 4,800 道）。
  - 四个层级：原子识别、片段内复合理解、跨片段因果推理、高层战术抽象。
- **结果**（零样本整体准确率）：

  | 模型 | 零样本 | 加 TacticCheck |
  |---|---|---|
  | Qwen2.5-VL | 27.57% | 35.20% |
  | Qwen3-VL | 29.00% | 37.20% |
  | VideoLLaMA3 | 32.30% | 39.43% |
  | Molmo2 | 31.31% | 39.14% |

  TacticCheck 是一种测试时自检重排方法，不更新参数。主要失败点在空间状态、"谁和谁配合"的交互绑定，以及比分细节。
- **接入 RallyIQ**：
  - 这是反面证据：**不要让 VLM 直接看视频做战术判断。**
  - TacticCheck 的思路可以借鉴到 RallyIQ 的落地校验：先让 LLM 回答一份低层事实清单（谁击球、在哪个区、比分），与结构化数据比对，一致后再输出高层战术结论。

### 14. Server Quality Score（SQS）：网球统一发球质量指标（2026）

- **论文**：https://arxiv.org/abs/2602.08083 （Li, Balajee, Wieand, Pipping-Gamón，2026-04）。
- **许可证**：正文说"代码在项目 GitHub 仓库"，但**没有给出链接**，许可证未知。数据来自 Jeff Sackmann 的公开逐分数据。
- **方法**：
  - 一发、二发分别拟合 logistic 混合效应模型。
  - 结果变量：发球方是否在 3 拍内拿下这一分。
  - 固定效应：平均发球速度、速度波动、主要落点、落点熵。
  - **发球方和接发方作为交叉随机截距**。SQS 定义为"面对平均水平接发方时"的线性预测值。
- **结果**：在温网男单上，与一发效率的相关 r = 0.667，加权 Elo 只有 0.148。
- **接入 RallyIQ**：
  - 业余球馆里，同一批人会和很多不同的对手打，正好适合用"球员 × 对手交叉随机效应"把**对手强度**从指标里剥离。
  - 例如"你的发球前 3 拍得分率"可以给出两个版本：对手调整后的和原始的。
  - 可以用 statsmodels 或 PyMC 自己实现，不涉及许可问题。

### 15. EXSPRT 序贯胜率预测（Jo，BMC SSMR 2025）

- **论文**：Jo, *BMC Sports Science, Medicine and Rehabilitation* 17:48（2025）。https://d-nb.info/1366965779/34 。文章许可为 CC BY-NC-ND 4.0。没有代码。
- **方法**：
  - 专家系统序贯概率比检验（EXSPRT）。
  - 事件难度参数从 2018 年 BWF 男单 100 场、7,871 分编码得到。
  - 先验胜率取自博彩赔率。
- **结果**：六个模型中最好的整体准确率为 0.730（三分类：平 / 胜 / 负）。
- **接入 RallyIQ**：价值有限。只能借鉴一个可视化：比赛进程中的实时胜率曲线。业余比赛没有赔率先验，可以改用条目 11 / 14 的球员强度估计。

### 16. ShuttleArena：可解释的物理羽毛球自博弈（2026）

- **论文**：https://arxiv.org/abs/2608.25246 （Peize Ding）。
- **代码**：https://github.com/pd2714/RL_badminton ，**MIT**（LICENSE 原文为"Copyright (c) 2026 Peize Ding"）。
- **内容**：
  - 物理飞行模型加 PPO 自博弈。
  - 核心观点：**一拍的价值取决于击球后能不能及时回位，覆盖对手的回球。**
  - 用 ShuttleSet22（43,994 拍）做了一致性检查。结果是模拟回合偏短、回位点偏前。
  - 作者明确说**没有验证对业余球员的泛化性**。
- **接入 RallyIQ**：
  - 优先级低，借鉴一个概念：在逐拍价值里加入"击球后 0.5–1 秒内的回位距离和时间"作为特征，在报告里体现为"杀完不回位"一类问题。
  - 这一点也对应 TactiPlay 的"站位与回位"问题域。

### 17. Bridging the Gap：把单打训练的姿态击球识别迁移到双打（2025）

- **论文**：https://arxiv.org/abs/2508.13507 （Baek & Yun，崇实大学，2025-08）。
- **代码**：https://github.com/100-heon/badminton_double_analysis 。**没有 LICENSE 文件**：API 返回 null，main 和 master 分支的 LICENSE* 均为 404。按**保留所有权利**处理。
- **方法**：
  - YOLOv11x + BoT-SORT 跟踪，加上一个基于速度的 ID 重分配步骤。
  - ViT-Pose 输出 COCO-17 关键点。
  - ST-GCN 编码器用对比学习（NT-Xent）得到 64 维嵌入，前场和后场分别使用独立的骨干。
  - Transformer 做"击球 / 非击球"二分类。
- **结果**：

  | 场景 | 本文 F1 | YOLOv11x 视觉基线 F1 |
  |---|---|---|
  | 单打 | 0.8975 | 0.9877 |
  | 双打 | **0.7104**（准确率 0.6686） | 0.5899 |

  双打只评测了两场比赛。
- **接入 RallyIQ**：
  - 双打的击球人归属可以加入"每名球员的姿态击球概率"特征，和第 1 节的手腕速度峰值配合使用。
  - 代码不能用。纯姿态的上限也再次说明它只能做辅助。

### 18. 两篇 2026 综述（方法论参考）

- **Zhang & Guo，《From prediction to execution》**，*Frontiers in Physiology* 2026。https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2026.1989299/full
  - 对 66 项羽毛球战术建模研究编码，结论是：
    - **没有任何研究用比赛中实测的球员状态验证模型**，也没有研究记录教练按模型建议执行后的结果。
    - 大多数"预测"研究衡量的是"和历史选择的一致程度"，而不是战术是否有效。
  - 对 RallyIQ 的启示：报告里要区分"描述 / 预测 / 处方"。处方类建议要写明执行前提（体能、技术），并提供回访机制，即下一场比赛检验建议是否被执行、是否有效。
- **Arthur 等，《Making the World's Fastest Racket Sport even Better》**，Research Square 预印本，2026-03，**CC BY 4.0**。https://www.researchsquare.com/article/rs-9105158/v1
  - 综述了 2018–2025 年 51 项研究。结论是证据集中在精英和大学生运动员，业余和青少年样本很少。缺乏共享的验证集和可重复性检验。

---

## 3. 商业球拍运动产品向用户展示的指标

| 产品 | 运动 | 采集方式 | 展示给用户的指标 | AI 教练 / 对手分析 | 价格（以页面为准） |
|---|---|---|---|---|---|
| **SwingVision** | 网球 / 匹克球 / 板式网球 | iPhone/iPad 架在底线后，60 fps；可 1–2 台。ITF PAT-25-037 于 2025-10-31 获批，报告**不含精度数据** | 击球速度、深度、准确度（落点）、回合长度；落点和击球点热力图；发球速度和旋转（可过滤 Kick）；实时与语音司线、自动计分；按拍型过滤、球迹动画、目标区练习。ITF 报告列出：击球/过网/弹地 3D 位置、正反手和发球、上旋和切削、球速 | 每次训练后给个性化建议，"AI Coach"，周目标；USTA NTRP/WTN 与 DUPR 对接 | App Store 页面列 Pro 年费 299.99 / 月费 39.99，Plus、Max 档另计；描述中另写 179.99 美元/年，各国不同 |
| **PB Vision** | 匹克球 | 手机三脚架，离地 ≥ 4 英尺，四个角都要在画面内，720p | 统计与技能评级、集锦；发球速度和深度、接发统计；第三拍分类（吊 / 混合 / 抽 / 挑）；抢网、手速对攻、最长回合、场地覆盖；总览、队伍、个人三种视图；Shot Explorer | "问 AI Coach 下一步练什么，它会找出体现你强项和弱项的片段"；可导出数据给外部 LLM；与 DUPR 合作做 AI 评级 | 首场免费；Starter 19.99 / 99.99，Premium 49.99 / 399；另有分钟包和 AI 教练点数 |
| **PointIQ**（Vibral，域名 rallyiq.pages.dev） | 匹克球 | 导入 PB Vision 的 JSON | 胜率、DUPR、击球准确率；按拍型、结果、位置回放 | **Matchups：针对某一对对手的双打 scouting**；搭档默契分、教练简报、训练计划 | 未列 |
| **ATP / TennisViz INSIGHTS**（Tennis Data Innovations） | 职业网球 | 赛事跟踪数据 | 进攻占比（In Attack）、Conversion / Steal Score、**Shot Quality（0–10）**、打法风格。底层要素：60 多种拍型、局面（发球 / 接发 / 双底线 / 网前）、阶段（进攻 / 相持 / 防守）、7 种战术和 48 种套路 | 以巡回赛和赛事均值作为对照 | 面向媒体 |
| **PlaySight** | 网球智能球场 | 场馆固定多机位 | 比赛分析、发球分析与热力图、底线球分析、接发分析、Game Chain | — | B2B |
| **Zepp Tennis** | 网球 | 拍柄传感器（Zepp Tennis 2） | 球速、旋转、甜区一致性；正反手、截击、高压、发球，并细分上旋 / 平击 / 切削；击球计数 | Smart Rally 视频 | 需要硬件 |
| **流行羽"翎析AI"**（与百度智能云合作，2026-07 用于青少年业余巡回赛） | 羽毛球 | 场馆多机位 AI 摄像头 | "TOP 球速、步法效率、得失分热区分析"；动作识别 | 每人一份 AI 评估报告和 15 秒集锦，"针对短板给出日常训练建议" | 赛事配套 |
| **多练**（duolian.cc） | 乒乓球 / 羽毛球 | 手机视频导入 | 自动切回合、局和人；回合数、有效比赛时长。**点头像记录每分谁赢**后，自动算出全场得分率、分局走势、最长连续得分、丢分后的得分率变化、短 / 中 / 长回合得分率 | — | 会员每周 4 小时智能切分，另售时长包 |
| **好球**（苏州好球科技）/ **球迹**（宁波坤达） | 羽毛球 | Apple Watch 戴在持拍手 | 挥拍总数、最大挥速、杀球数、正反手和上下手拍型；心率和卡路里；排行榜 | — | ¥15–18/月，¥38/年，¥68 终身 |
| **Smashspeed** | 羽毛球 | 手机录杀球 | 羽毛球峰值速度、下压角度、逐帧速度、一致性；全球排行榜 | — | 免费加内购 |
| **Badminton Analyst** | 羽毛球 | 2 分钟视频加一个提问 | 步法、平衡、站位、覆盖、战术倾向 | LLM 式文字点评（优点 / 改进 / 建议） | €2.99/月 |

另有两类传感器产品，见虎嗅 2026-02 转载 AING 硬迹的文章：
- 拍柄外挂传感器 USENSE，记录挥速、拍型、卡路里。
- 2025 年李宁与香港理工大学合作的"离线强化学习智能球拍"，宣称能给出实时战术建议，细节没有公开。

**对 RallyIQ 的启示**：
1. **业余羽毛球产品的指标空位**：现有产品给的是计数类和速度类指标（球速、挥拍数、热区、回合得分率），还没有"逐拍价值"和"针对具体对手的弱点"。网球职业端的 Shot Quality 和 Conversion/Steal，以及匹克球的 PointIQ Matchups，证明用户愿意为这类指标付费。
2. **可以照搬的低成本设计**：
   - 多练的"点头像记每分胜者"：几乎零成本拿到 xPoint 的训练标签，并能立刻产出"短 / 中 / 长回合得分率""丢分后得分率"。
   - PB Vision 的"AI 教练自动找片段"：每条结论附证据片段。
3. **类 ATP 的进攻 / 防守阶段指标**，RallyIQ 可以这样定义：
   - 阶段判定：用击球高度（过网高点以上 / 以下）、拍型（杀、扑、推为进攻；挑、接杀为防守）、对手位置。
   - Conversion：进攻阶段赢下回合的比率。
   - Steal：防守阶段赢下回合的比率。
4. **精度要诚实**：SwingVision 的 ITF 认证也不包含精度数据。RallyIQ 可以把"每项指标的置信度和样本量"直接展示给用户，作为可信度卖点。

---

## 4. 建议的"如何战胜这名对手"分析层（把上面的条目串起来）

**输入**：RallyIQ 现有输出，包括击球时刻和击球人、拍型概率、双方脚点场地坐标、羽毛球轨迹与落点区、回合起止，以及每分胜者（rally.md 的发球锚定层推出，或用户点选补标）。

**第 1 步：逐拍状态表**。每一拍一行，字段如下：
- 击球方和接球方的场地坐标，用 3×3 或 2×3 场区编码。
- 击球高度：上手 / 平 / 下手，由击球帧的球高相对肩、头关键点判断。
- 来球拍型和本拍拍型（概率向量）。
- 距上一拍的时间。
- 本拍落点区（如可得）。
- 击球后 0.6 秒内的回位距离（条目 16）。
- 接球方失位距离（条目 12）。
- 比分状态、回合内拍序。

**第 2 步：逐拍价值**（条目 1、2、9、10）：
1. 用 GBM 输出 P(制胜) 和 P(失误)，拆成 P(不失误) × P(对手接不回)。先在 ShuttleSet 上预训练（只用 RallyIQ 能测到的特征），再用业余视频按每分胜者做保序回归校准。
2. 单拍价值 = 本拍与对手上一拍之间 P(本方赢下回合) 的变化量；也可以用 Bradley-Terry 逐拍奖励（MIT 代码可参考）。
3. 不确定性传递：击球漏检（目前召回约 59%）和拍型误差（约 77% 准确）会传到价值估计里。用拍型**概率加权**汇总；回合里有缺失拍时，标注"序列不完整"并降低这个回合的权重。

**第 3 步：对手画像**（条目 4、7、11、14）：
- **弱区**：按"对手击球区 × 来球类型"汇总对手的平均单拍价值和失误率，用 Wilson 置信区间，样本少于 8 拍的单元格不报告。
- **倾向与可预测性**：估计 P(对手出球拍型和方向 | 击球区, 来球)，用 Dirichlet-多项式模型向全体业余人群先验收缩，同时报告熵。熵低就写"可预判"。
- **回位短板**：哪类击球之后对手的回位时间最长，或失位距离最大。
- **对手强度调整**：同一对手有多场数据时，用球员 × 对手交叉随机效应拟合（参考 SQS）。
- **双打**：用 CCEI 和 TLAI 找出覆盖漏洞和"承受威胁更多的一方"（条目 6）。

**第 4 步：LLM 报告**（条目 3、13、18）：
- 输出 schema 采用 TactiPlay 的四个主题和三个问题域，每条结论必须带四样东西：事件 ID 列表、计数、置信区间、视频片段。
- 落地校验分两层：
  1. 先让 LLM 复述低层事实（谁击球、在哪个区、比分），与结构化数据比对（借鉴 TacticCheck）。
  2. 再核验高层结论引用的数字是否与统计表一致。
- 区分"描述 / 预测 / 建议"三类语句。建议类语句只在样本充分时给出，并写明执行前提。
- "如何战胜他"部分最多给 3 条，按"预期收益 × 样本可信度"排序。示例："把球压到他反手后场（N = 23，他在此区单拍价值 −0.12，失误率 30%，95% CI 18–45%），随后准备封网前（他在该区 68% 回直线网前）。"

---

## 5. 许可证汇总

| 条目 | 代码 | 权重 / 数据 | 商用结论 |
|---|---|---|---|
| sarah-agrawal/rallyiq | 无 LICENSE，保留所有权利 | 使用 ShuttleSet（仓库根 MIT） | 只借鉴思路，自己重写 |
| Offline RL for Badminton（PolyU） | **MIT** | 仓库内含 ShuttleSet 衍生数据 | 可用（注意 ShuttleSet 来源） |
| TactiPlay | 未发布 | 未发布 | 只借鉴设计 |
| RallyTemPose | **MIT** | README 未提权重 | 可用 |
| CoachAI+ / RallyNetv2 | 无 LICENSE，保留所有权利 | — | 只借鉴思路 |
| RallyNet（CoachAI-Projects 内） | **MIT**（仓库根） | 示例数据 | 可用 |
| ShuttleSet / ShuttleSet22 | 仓库根 LICENSE 为 MIT；ShuttleSet 子目录 README 没有单独的许可声明 | 原始视频版权属 BWF，不在仓库内 | 标注 CSV 可按 MIT 使用；视频不可 |
| DShuttleNet | 没有代码 | Kaggle，**许可证未核实** | 核实前只用于设计指标 |
| BadmintonVision | **MIT**（附注：不含转播视频） | 数据需申请；文章为 CC BY-NC-ND 4.0 | 代码可用 |
| MT-HTA | 无 LICENSE，保留所有权利 | 网球部分用 MCP（CC BY-NC-SA 4.0） | 只借鉴思路 |
| ICAART 2026 博弈树 | 没有代码 | — | 按公式自己实现 |
| ShuttleEnv | 没有代码 | Lin-Lee 数据未发布 | 只借鉴思路 |
| "How good was my shot?" | 没有代码 | 未发布 | 只借鉴思路 |
| Court to Conversation | 没有代码 | 只有 Zenodo 演示视频 | 只借鉴思路 |
| DoublesEval | 文中称有 GitHub，但无链接 | — | 未核实 |
| SQS（网球） | 文中称有 GitHub，但无链接 | Sackmann 数据 | 方法自己实现 |
| ShuttleArena | **MIT** | — | 可用（优先级低） |
| Bridging the Gap（双打） | 无 LICENSE，保留所有权利 | — | 只借鉴思路 |
| arXiv 2606.15992 | 未发布 | 视频不能共享 | 按论文描述自己实现 |
| Match Charting Project（网球逐拍数据） | — | **CC BY-NC-SA 4.0**，作者明确强调禁止商用 | **不能用于商用训练** |

---

## 6. 局限与未核实项

- DShuttleNet 的 Kaggle 许可证：kaggle.com 被代理拦截，抓到的页面里没有许可证字段。
- ICAART（SciTePress）论文的许可证：页面没有声明。
- KBS 论文只看到截断的摘要和正文，没看到 RAG 评测的数字。
- DoublesEval 和 SQS 都声称代码在 GitHub 上，但没有给出链接，所以没能核实。
- TactiPlay 提到的两款中国商业工具"Shanyu AI"和"Yuji"，没能找到对应的官方页面。只找到同类的"球迹""好球"手表 App，二者可能不是同一产品。
- 虎嗅和 tixiaolu 的原文没能抓取，只看到二手摘要。
- CoachAI+ 的发表年份：PDF 编号是 20251430，SSAC 页面套的是 2026 年的模板，推测为 SSAC 2025，未最终确认。
- 文中的数字由网页抓取工具从原文提取，个别可能有转录误差，例如 DShuttleNet 的 MoCVAE 交叉熵单元格在原文里就是乱码。正式引用前请回原文核对。
- 本次没有在 RallyIQ 的业余视频上实际运行任何模型。业余场景下的效果全部是推断。

---

## 来源

- [arXiv 2606.15992 摘要页](https://arxiv.org/abs/2606.15992) ・ [v2 HTML](https://arxiv.org/html/2606.15992v2) ・ [v1 HTML（CC BY 4.0）](https://arxiv.org/html/2606.15992)
- [sarah-agrawal/rallyiq](https://github.com/sarah-agrawal/rallyiq)
- [Offline RL for Badminton 论文 PDF](https://wenminggong.github.io/papers/Offline_RL_for_Badminton_EAAI_paper.pdf) ・ [代码](https://github.com/Wenminggong/Offline_RL_for_Badminton) ・ [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0952197625034268)
- [TactiPlay arXiv 2607.27125](https://arxiv.org/abs/2607.27125)
- [RallyTemPose 论文](https://openaccess.thecvf.com/content/CVPR2024W/CVsports/papers/Ibh_A_Stroke_of_Genius_Predicting_the_Next_Move_in_Badminton_CVPRW_2024_paper.pdf) ・ [代码](https://github.com/MagnusPetersenIbh/RallyTemPose)
- [CoachAI+ PDF](https://cdn.prod.website-files.com/5f1af76ed86d6771ad48324b/6791fce2ea6c69e3fbc7b5db_CoachAI%2B%20Badminton%20Environment.pdf) ・ [SSAC 页面](https://www.sloansportsconference.com/research-papers/coachai-badminton-environment-realistic-badminton-game-simulator-for-enhancing-player-performance) ・ [CoachAI-Plus 仓库](https://github.com/KuangDW/CoachAI-Plus) ・ [CoachAI-Projects](https://github.com/wywyWang/CoachAI-Projects) ・ [RallyNet arXiv](https://arxiv.org/abs/2403.12406)
- [DShuttleNet arXiv 2610.14016](https://arxiv.org/abs/2610.14016) ・ [Kaggle 数据](https://www.kaggle.com/datasets/anonymousay58dki/dshuttlenet/versions/7)
- [BadmintonVision PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC13595539/) ・ [代码](https://github.com/liangyongjie51/BadmintonVision)
- [MT-HTA arXiv 2611.15529](https://arxiv.org/abs/2611.15529) ・ [代码](https://github.com/gndd1221/For-Sports-event-modeling-with-rally-and-score-hierarchical-A-Multi-Task-Hierarchical-Transformer)
- [ICAART 2026 最优击球预测](https://www.scitepress.org/publishedPapers/2026/143176/pdf/index.html)
- [ShuttleEnv arXiv 2603.17324](https://arxiv.org/abs/2603.17324)
- [How good was my shot? arXiv 2603.25736](https://arxiv.org/abs/2603.25736)
- [Court to conversation（KBS）](https://www.sciencedirect.com/science/article/abs/pii/S0950705125020659)
- [DoublesEval arXiv 2608.24439](https://arxiv.org/html/2608.24439)
- [SQS arXiv 2602.08083](https://arxiv.org/html/2602.08083)
- [Jo 2025 EXSPRT](https://d-nb.info/1366965779/34)
- [ShuttleArena arXiv 2608.25246](https://arxiv.org/abs/2608.25246) ・ [代码](https://github.com/pd2714/RL_badminton)
- [Bridging the Gap arXiv 2508.13507](https://arxiv.org/html/2508.13507v1)
- [Frontiers 2026 综述](https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2026.1989299/full) ・ [Research Square 系统综述](https://www.researchsquare.com/article/rs-9105158/v1)
- [Match Charting Project](https://github.com/JeffSackmann/tennis_MatchChartingProject)
- [SwingVision App Store](https://apps.apple.com/au/app/swing-tennis/id989461317) ・ [ITF PAT 报告](https://www.world.tennis/media/15294/pat-25-037-approval-report-swingvision.pdf)
- [PB Vision App Store](https://apps.apple.com/app/id6467020610) ・ [Pickleball Union 介绍](https://pickleballunion.com/pb-vision-pickleball-ai/) ・ [PointIQ](https://rallyiq.pages.dev/)
- [ATP INSIGHTS 介绍](https://www.atptour.com/en/news/insights-introduction)
- [PlaySight 网球](https://playsight.com/?p=115079) ・ [Zepp Tennis](https://apps.apple.com/us/app/zepp-tennis/id1248208481)
- [流行羽翎析AI（中国日报）](https://cn.chinadaily.com.cn/a/202607/20/WS6a5db434a310d709c2fbe719.html) ・ [多练](https://duolian.cc/) ・ [好球 App](https://apps.apple.com/cn/app/%E5%A5%BD%E7%90%83-%E4%B8%93%E4%B8%9A%E7%BE%BD%E6%AF%9B%E7%90%83%E8%BF%90%E5%8A%A8%E5%88%86%E6%9E%90/id1622183934) ・ [球迹 App](https://apps.apple.com/cn/app/%E7%90%83%E8%BF%B9-%E7%BE%BD%E6%AF%9B%E7%90%83%E8%BF%90%E5%8A%A8%E6%95%B0%E6%8D%AE%E8%AE%B0%E5%BD%95%E5%92%8C%E5%88%86%E6%9E%90/id6529541519) ・ [Smashspeed](https://apps.apple.com/app/6748543435) ・ [Badminton Analyst](https://apps.apple.com/app/id6792478568) ・ [虎嗅（AING 硬迹转载）](https://m.huxiu.com/article/4833479.html)
- [VON CRAMM（网球 Shot IQ，2020，背景参考）](https://arxiv.org/abs/2005.12853)
