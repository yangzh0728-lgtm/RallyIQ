# strokes：击球类型分类（stage `strokes`）

入口：`rallyiq.strokes.stage:run(ctx) -> StrokesResult`。读取 `ctx.events`（原始击球）、`ctx.shuttle`、
`ctx.players`，为每一拍输出一个 `StrokeInfo`：类型、置信度、前几名候选、`status`、复核原因和测得的证据。
API 把它按 `shot_id` 合并进 `GET /api/videos/{id}/events`（`shot_type` + `stroke`），人工用
`set_shot_type` 修正（追加写入，原始预测保留，击球本身的状态不受影响）。

## 类型（12 类）

与 ShuttleSet 合并后的 25 类（"未知" + 上/下半场 × 12）一致，去掉半场前缀，方便以后直接接 ShuttleSet 训练的模型：

| code | 中文 | code | 中文 |
| --- | --- | --- | --- |
| `short_serve` | 发短球 | `net_shot` | 放网 |
| `long_serve` | 发高远球 | `cross_net` | 勾对角 |
| `clear` | 高远球 | `push` | 推球 |
| `smash` | 杀球 | `net_kill` | 扑球 |
| `drop` | 吊球（含切吊） | `drive` | 平抽 |
| `lift` | 挑球 | `block` | 挡网（接杀挡） |

`shot_type = null` 表示这段视频上不可判定，不是"没有类型"。

## 默认分类器：`rules_v1`（无需权重）

`features.py` 从已有数据测量每一拍的证据，测不到的就是 `null` 并附原因：

| 证据 | 含义 |
| --- | --- |
| `contact_h` | 击球瞬间球的高度，以击球者"脚踝→鼻子"的像素长度为 1；约 1.6 为头顶上方击球，< 0.6 为低手 |
| `hitter_depth_m` / `receiver_depth_m` | 击球者 / 下一拍接球者离网的距离（米，需要球场标定） |
| `court_speed_mps` | 击球者到接球者的水平距离 ÷ 飞行时间 |
| `flight_ms` | 到下一拍（或最后一拍到落地）的时间；没有标定时用它代替速度 |
| `rise` | 飞行弧线比两端中较高一端高出多少，以那一端球员（通常是远端球员）的身高为单位，透视基本抵消 |
| `left_frame` | 飞行中球飞出画面上沿（很高的球） |
| `lateral_m` | 击球者与接球者的横向距离（勾对角） |

`rules.py` 把每种类型写成这些证据的模糊"与"（0–1 斜坡），测不到的证据按 0.5 处理，
归一化后得到 12 类的分布，取前三作为候选。首拍按发球处理；上一拍是杀球时提高"挡网"的分数。

## 复核策略（任何后端都一样，`classify.py`）

- 可测证据（击球点高度、击球者位置、接球者位置、速度/时间、弧线）不足 30%：`undeterminable`，`shot_type = null`。
- `ok` 需要同时满足：置信度 ≥ 0.6、第一名比第二名高 ≥ 0.2、证据至少 80% 可测、击球本身是 `ok` 且击球者已知。
- 其余为 `needs_review`，原因码：

| code | 含义 |
| --- | --- |
| `ambiguous_types` | 前两种类型接近 |
| `incomplete_evidence` / `insufficient_evidence` | 证据不完整 / 不足 |
| `hit_needs_review` | 击球本身待复核 |
| `no_court_position` | 没有球场坐标（标定失败） |
| `no_hitter_pose` / `hitter_unknown` / `no_shuttle_at_hit` | 缺骨架 / 击球者 / 击球点的球 |
| `last_shot_of_rally` | 回合最后一拍，没有接球方（不单独阻止 `ok`） |
| `next_hit_same_side` | 下一拍在同一半场，中间可能漏检 |
| `flight_not_observed` | 飞行段几乎没有球的观测 |
| `first_hit_not_serve_like` | 首拍不像发球，真正的发球可能漏检 |
| `no_rule_fits` | 没有任何规则匹配 |

## 合成基准

`synthetic.py`：三维物理（二次阻力，终端速度 6.8 m/s）+ 1.75 m COCO-17 骨架 + 后场高位针孔相机，
按真实的击球衔接（如 高远 → 吊/杀，杀 → 挡/挑/抽）生成单打回合，并给出真值类型。8 个种子 × 8 回合（约 320 拍）：

| 条件 | 准确率 | 标为"可用"的占比 | "可用"中正确 |
| --- | --- | --- | --- |
| 干净 | 0.88 | 77% | 0.94 |
| 15% 丢球 | 0.87 | 77% | 0.94 |
| 击球帧 ±2 帧误差 | 0.85 | 78% | 0.94 |
| 3 px 噪声 + 10% 丢球 + ±1 帧 | 0.87 | 80% | 0.94 |
| 60 fps | 0.87 | 82% | 0.95 |
| 无球场标定 | 0.65 | 0%（全部待复核） | — |

最难分的是 推球/平抽、挡网/平抽、放网/勾对角。**这些数字只说明规则按设计工作，不代表真实视频上的精度**；
需要在 `evaluation/` 里加入真实击球类型标注后再调阈值。

## 可选：scikit-learn 分类器（`sklearn`）

`learned.py` 用 `HistGradientBoostingClassifier` 在**同一组证据**（上表的特征 + 回合中的位置）上学习类型，
测不到的证据保持 NaN，不填猜测值。复核策略仍是上面那一套。安装：`pip install -e .[strokes-ml]`。

```bash
# 合成数据快速检查（不代表真实视频精度）
python scripts/learn_strokes.py --source synthetic --out ~/rallyiq-data/models/strokes_synth.joblib
# ShuttleSet 训练片段（scripts/build_train_set.py 已跑过管线，只读缓存产物；可给多个文件夹）
python scripts/learn_strokes.py --source shuttleset \
    --data ~/rallyiq-data/public/shuttleset_train ~/rallyiq-data/public/shuttleset_train_v3 \
    --out ~/rallyiq-data/models/strokes.joblib   # 默认 --features v2
# 在留出的比赛上评估
python scripts/learn_strokes.py --source shuttleset --data ~/rallyiq-data/public/shuttleset_test \
    --eval ~/rallyiq-data/models/strokes.joblib
```

评估用 `GroupKFold`（按比赛分折，同一场比赛不会同时出现在训练和验证里），输出准确率、宏平均 F1、
每类精确率/召回率和混淆矩阵，并在同一批击球上给出 `rules_v1` 的准确率作对比。
管线里启用：`{"strokes": {"classifier": "sklearn", "sklearn": {"model": "<path>.joblib"}}}`，
模型缺失时退回规则并告警（`require_model: true` 则报错）。joblib 本质是 pickle，只加载自己训练的模型。

合成数据 16 段视频（658 拍）上：规则 0.88，sklearn 交叉验证 0.93（宏 F1 0.92），
主要混淆仍是 推球/平抽、放网/勾对角。同样只说明流程可用，真实精度要用 ShuttleSet 跑出来。

### 特征集 v2（`context.py`）和真实视频上的结果

`--features v2`（默认）在上面的证据之外再加 18 个特征，都来自管线已有的结果，不增加计算：

- **回合上下文**：来球（上一拍的飞行时间、弧高、对手的击球高度、站位深度）和回球（下一拍的击球高度、飞行时间）。挡网接杀球、挑球接放网，这些靠单拍本身分不出来。
- **方向**：击球方和接球方的左右位置（以击球方自己的左右为准）以及是否斜线（`cross < 0`）。
- **姿态和出球**：最高手腕和手肘相对头部的高度、手腕离身体的距离，以及击球前后 200 ms 内羽毛球在画面里的速度（以击球者身高为单位）。

ShuttleSet 结果（2026-11-07；训练 = 38 场里已跑完的 59 段、915 拍；测试 = 整场留出的 6 场、184 拍；
只统计时间上对上标注的击球，±3 帧）：

| 方法 | 测试准确率 | 测试宏 F1 | 训练集按比赛交叉验证 |
| --- | --- | --- | --- |
| 规则 `rules_v1` | 45.7% | — | 42.4% |
| sklearn，基础特征 | 56.0% | 0.43 | 64.5% |
| sklearn，特征 v2 | **63.6%** | **0.51** | 67.4% |

经过完整 `strokes` 阶段（含复核策略）时 v2 在测试集上是 62.5%。测试只有 184 拍，±3.5 个百分点以内的差别不可靠；
交叉验证和测试两边 v2 都更好。去掉任何一组特征（方向、来球、回球、出球速度）测试准确率都会下降；姿态组单独加效果不明显，
保留是因为交叉验证上有帮助。仍然最难的是 推球/挑球、平抽/挡网、放网/勾对角。旧模型（基础特征）照样能加载。

### 默认模型（随仓库发布）

`models/strokes_v2.json` 是用 85 段训练片段（19 场、1270 拍）训练好的 v2 模型（100 棵树 × 12 类，约 1 MB），
存成 JSON 树，用 numpy 计算：**不需要 scikit-learn，也不用 pickle**。`strokes` 阶段默认
`classifier: "auto"` 就用它；文件缺失或损坏时退回规则。要用规则：`{"strokes": {"classifier": "rules"}}`。

| 默认分类器 | 6 场留出比赛（184 拍） |
| --- | --- |
| 之前：规则 `rules_v1` | 45.7% |
| 现在：`gbt_v2`（经过完整 `strokes` 阶段） | **63.0%**（模型本身 65.2%；按比赛交叉验证 68.0%） |

重新训练并替换：`python scripts/learn_strokes.py --source shuttleset --data <训练文件夹...> --out rallyiq/strokes/models/strokes_v2.json`
（`--out` 以 `.json` 结尾就写 JSON，否则写 joblib）。注意：只在 BWF 转播单打上训练和测试过，
你自己拍的视频（侧面固定机位、业余比赛）上的精度还没有标注数据可测。

另外试过两种扩展，都没有提升，没有采用：把击球前后的挥拍姿态序列加进特征（测试 61–63%），
以及用前后拍的预测类型做第二阶段（测试 60.9%）。

## 可选：BST（骨架 + 羽毛球 Transformer）

`bst.py` 按作者的数据预处理（"两拍之间"片段、`seq_len` 100、骨架按人框归一化并居中、
位置按球场归一化、球按分辨率归一化）组织输入，把 25 类输出按击球者半场折叠到上面 12 类：

```python
ctx.config["strokes"] = {
    "classifier": "bst",
    "bst": {"repo": "~/src/BST-Badminton-Stroke-type-Transformer",
            "weights": "~/weights/bst/<ShuttleSet 25 类、seq_len 100 的权重>.pt",
            "model_name": "BST_CG_AP", "pose_style": "JnB_bone"},
}
```

需要 torch 和作者仓库的代码；缺任何一项时回退到 `rules_v1` 并记日志（`require_model: true` 则直接报错）。

**许可证**：BST 代码是 MIT。作者公开的权重放在 Google Drive，没有单独的许可证，训练数据 ShuttleSet 的标注是 MIT，
但视频本身是 BWF 比赛转播。按 README 的依赖准入规则，这些权重目前只能在本机做个人研究/对比用，
**不要随仓库分发，也不要用在任何商用场景**；要商用需要用自己授权的数据重新训练。仓库不会自动下载它们。

输入准备（numpy）有单元测试；模型本身没有在 CI 中运行（CI 没有 torch 和权重），
第一次在本机接上权重时需要用几段已标注的片段核对输出。

## 配置（`ctx.config["strokes"]`）

`classifier`、`ok_threshold`、`ok_margin`、`min_evidence_ok`、`min_evidence`、`min_rule_score`、`top_k`、
`min_candidate_p`、`require_model`、`bst`。规则斜坡参数在 `rules.RulesConfig`，特征参数在 `features.FeatureConfig`。
