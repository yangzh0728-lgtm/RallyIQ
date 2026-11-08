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

`models/strokes_v3.json` 是 v3 特征的模型（300 轮 × 12 类、每棵最多 31 个叶子，约 4 MB），存成 JSON 树，用 numpy 计算：
**不需要 scikit-learn，也不用 pickle**。`strokes` 阶段默认 `classifier: "auto"` 就用它；文件缺失或损坏时退回规则。
要用规则：`{"strokes": {"classifier": "rules"}}`。

训练数据：所有已建好的训练片段（34 场比赛；同一段在两个管线版本下的产物都用上，共 2486 拍），
在另外 6 场留出比赛（189 拍）上测：

| 默认分类器 | 模型本身 | 经过完整 `strokes` 阶段 | 按比赛交叉验证 |
| --- | --- | --- | --- |
| 规则 `rules_v1` | 46.0% | — | — |
| `gbt_v2`（19 场训练） | 65.1% | 63.0% | 68.0% |
| `gbt_v2` 特征，34 场训练 | 67.2% | 65.6% | 71.2% |
| v3：再加飞行弧线形状（`trajectory.py`） | 68.3% | 66.1% | 71.7% |
| **`gbt_v3`（现在）**：v3 + 交叉验证选出的参数 | **70.9%** | **69.3%** | 73.1% |

v3 的弧线特征（弧高、最高点出现的时刻、弯曲度、起飞爬升速度、末段/初段速度比、轨迹覆盖率）是冲着
“推球 vs 挑球”加的（这是最大的混淆：各有约一百拍互相错判）：推球 F1 0.32 → 0.36，挑球 0.76 → 0.80。
提升不大，主要的增益来自更多比赛的数据。

参数（`TrainConfig`：学习率 0.06、300 轮、31 个叶子、每叶至少 40 拍、不加类别权重）是只在 34 场训练比赛上
按比赛 5 折交叉验证、从 40 组参数里选出来的，选完才在留出比赛上测一次。数据多了以后，更多更大的树有用了；
类别权重不再提高准确率（宏平均 F1 基本不变：0.586 → 0.591）。

重新训练并替换：`python scripts/learn_strokes.py --source shuttleset --features v3 --data <训练文件夹...> --shuttleset <ShuttleSet/set> --out rallyiq/strokes/models/strokes_v3.json`
（`--out` 以 `.json` 结尾就写 JSON，否则写 joblib）。注意：只在 BWF 转播单打上训练和测试过，
你自己拍的视频（侧面固定机位、业余比赛）上的精度还没有标注数据可测。

另外试过两种扩展，都没有提升，没有采用：把击球前后的挥拍姿态序列加进特征（测试 61–63%），
以及用前后拍的预测类型做第二阶段（测试 60.9%）。

ShuttleSet 的 CSV 把“过渡切球”写成“過度切球”，之前的映射只认“過渡切球”，这类吊球被当成没有标签。
现在两种写法都映射到 `drop`，`learn_strokes.py --shuttleset` 直接从 CSV 读类型。按修正后的标签重测，
当时的默认模型在 6 场留出比赛（189 拍）上是 65.1%（上面的表都按修正后的标签测）。

## 分层描述（`detail.py`，`StrokeInfo.detail`）

教练不会只说“杀球”，而是说“正手直线杀球，击球点较高，出球快，属于主动进攻”。每一拍分好类型后，
`strokes` 阶段再加上几层描述，并且每层都写明来源（`StrokeAttr.source`）：

| 层 | 取值 | 来源 | 6 场留出比赛 |
| --- | --- | --- | --- |
| 大类 family | 高远球 杀球 吊球 平抽 挡网 推球 挑球 放网 扑球 发球 | 由类型合并 | — |
| 细分 subtype | 勾对角、发短球、发高远球 | 由类型 | — |
| 正反手 hand | 正手 / 反手 / 头顶 | 模型 `model` | 77.5%（平衡 75.3%，全猜正手 49.2%）；sam 的 DJI 低机位视频 82%（34 拍，不含发球） |
| 击球点 contact | 高（网上）/ 低（网下） | 模型 `model` | 82.7%（平衡 81.6%，全猜多数 60.2%） |
| 方向 direction | 直线 / 斜线 | 模型 `model` | 93.5%（平衡 92.8%，全猜多数 61.4%） |
| 快慢 pace | 快 / 中 / 慢（和同类球的常见速度比） | 按轨迹测量 `measured` | 没有标签可测 |
| 主动/被动 role | 主动进攻 / 相持 / 被动防守 | 规则推断 `rule` | 没有标签可测 |
| 意图 intent | 进攻 / 调动 / 控制 / 防守 / 过渡 | 规则推断 `rule` | 没有标签可测 |

三个模型头（`models/stroke_attrs.json`，同样是 numpy 计算的 JSON 树）用 ShuttleSet 的 `backhand`、
`aroundhead`、`hit_height` 列和击球点/落点坐标训练（34 场 2515 拍；19 场 1310 拍时是 75.4% / 83.8% / 93.5%），在另外 6 场比赛上测试（191 拍），
击球点和方向头的输入是 v2 特征 + 击球前后的姿态窗口 + 球和两只手腕相对身体的左右位置 + 从高点击球推出的持拍手。
正反手头只看手腕和球相对身体的左右位置、持拍手和击球点高度（`HAND_HEAD_FEATURES`）：用全部特征时它在转播画面上
不差，但在 sam 的 DJI 低机位视频上 34 拍只对 20 拍，错的几乎都是把正手判成反手，因为姿态窗口和球场位置这些特征随机位变化。
手腕的左右按球员所在半场定（近端背对镜头、远端面对镜头），比用两肩的先后顺序可靠：侧身时两肩重叠，顺序是噪声。
重新训练：`python scripts/learn_stroke_attrs.py --data <训练文件夹...> --test <留出文件夹>`。

**认不了、没有输出的**：点杀、劈杀、劈吊、搓球（旋转放网）、平快发球、偷后场发球。ShuttleSet 没有这些标签
（或者标签分不开：杀球 vs 点扣在留出比赛上 54%，等于瞎猜），只在视频上看不出拍面和切削。
“欺骗”意图同样没有输出。主动/被动和意图是按类型、击球点高低、上一拍类型推出来的教练式解读，
App 里用虚线标签显示，AI 教练也被告知这两项只作参考。正反手在 sam 的 DJI_0009 片段上用人工核对过击球时刻的 58 拍测过（正反手由人逐拍看图标注，发球和看不清的不计），其余各层只在 BWF 转播单打上测过。

人工改了类型（`set_shot_type`）时，正反手、方向、击球点保留，其余各层按新类型重新推出（前端 `retypeDetail` 与后端 `retype` 一致）。
关闭：`{"strokes": {"detail": false}}`。

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

`classifier`、`detail`（默认 true）、`ok_threshold`、`ok_margin`、`min_evidence_ok`、`min_evidence`、`min_rule_score`、`top_k`、
`min_candidate_p`、`require_model`、`bst`。规则斜坡参数在 `rules.RulesConfig`，特征参数在 `features.FeatureConfig`。
