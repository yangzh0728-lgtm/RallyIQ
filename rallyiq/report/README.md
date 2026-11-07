# 赛后报告与 AI 教练（`rallyiq/report`）

```
effective events + players + calibration ──► stats.compute_match_stats ──► MatchStats
                                                                     │
                                         coach.generate_coach_report ▼ (provider: anthropic | offline | 自定义)
                                                 grounding 校验 ──► CoachReport ──► export.render_markdown
```

## 统计（`stats.py`，不调用模型）

全部基于**修正后的**事件，每个指标都是 `Metric`（数值、单位、样本量 `n`、`status`、`confidence`、说明）。
算不出来的值为 `null`，不会填默认值。

| 类别 | 指标 key（P1/P2 前缀为球员） |
| --- | --- |
| 回合 | `match.rallies`、`match.rally_len.mean/median/max`、`match.rally_dur.mean_s/max_s`、`match.rally_len.buckets` |
| 数据质量 | `match.confirmed_shots_pct`、`match.confirmed_rallies_pct`、`match.winners_known` |
| 击球 | `Pn.shots`、`Pn.shot_share`、`Pn.hit.zones`（击球位置）、`Pn.shot_types`（击球类型） |
| 移动 | `Pn.move.distance_m`、`Pn.move.per_rally_m`、`Pn.move.per_min_m`、`Pn.move.tracked_pct` |
| 覆盖与站位 | `Pn.cover.area_m2`、`Pn.cover.ratio_pct`、`Pn.pos.depth_mean_m`、`Pn.pos.depth_zones`、`Pn.pos.lateral_zones` |
| 打法特征 | `Pn.style.*`：击球构成、进攻 / 被动占比、主动转化、连续进攻、抢攻、拉吊突击、防守反击、节奏、调动、直线比例、落点多样性、吻合度等（见下文“打法风格”） |
| 得失分 | `Pn.win.won/rate_pct`、`short_rate_pct`、`long_rate_pct`、`serve_rate_pct`、`receive_rate_pct`、`own_last_shot_pct`、`lost_own_last_shot_pct`；`Pn.lost.last_shot_types`（失分回合中本人最后一拍的击球类型） |

- 移动只统计回合内、非插值的足点；5 帧滑动中值去抖，速度 > 8 m/s 的跳变视为跟踪错误剔除，间隔 > 0.5 s 不连线。
- 站位按“本方半场”换算：深度 = 离自己底线的距离，左右按本人面向球网的视角，换边后仍可比较。
- 球场标定不是 `ok` 时，米制指标降为“待复核”。
- 胜负指标只用已标注胜者的回合（工作台修正面板里设置），没有标注时为“不可判定”。
- **击球类型**：读取 `ShotEvent.shot_type`。击球分类模块接入后填这个字段即可，统计和报告自动出现分布（`SHOT_TYPE_LABELS` 里登记中文名），无需改报告代码。

## 逐拍建议（`moments.py`，不调用模型）

单打里每一拍已知击球者和击球类型的球，取击球那一刻双方在球场上的站位（球员跟踪，米制），
再看对手接这一拍移动了多远（对手下一拍的击球位置）、怎么回的、这一分谁赢，用几条规则找出该换打法的球：

| `issue` | 规则 | 建议 |
| --- | --- | --- |
| `clear_to_rear` | 高远球/挑球，而对手已在后场（且对手直接进攻、丢分或几乎不用移动） | 离对手最远的网前角：后场/中场打吊球，网前放网 |
| `short_to_front` | 吊球/放网/勾对角/挡网，而对手已在网前 | 离对手最远的后场角：后场打高远球，中场推球，网前挑球 |
| `lift_from_front` | 在网前挑球被对手杀/扑 | 放网到离对手最远的网前角 |
| `attack_off_balance` | 在后场被动位置杀球/扑球/平抽，直接丢分 | 先打高远球到后场中路过渡 |
| `to_opponent` | 对手移动不到 1 米就接到，然后进攻或得分 | 离对手最远的角，用从击球位置打得到的球 |

结果存在 `MatchStats.moments`（`ShotMoment`：视频时间、第几回合第几拍、双方站位和区域、对手移动距离、
回球、结果、规则建议），每人最多 12 拍（按严重程度挑），按时间排序。发球不评；双打暂不支持。
区域名：击球者位置按本人视角，对手位置和目标区域按“从击球者这边看过去”的左右。

教练报告（`CoachReport.shot_advice`）为每一拍写一条 `ShotAdvice`：时间、回合、拍数和站位永远取自
`ShotMoment`，模型只写 `advice`/`why`；模型漏写或写了不符的时间时改用规则说明（`source = "rules"`）。
在“人物”里指定了哪位是“我”（`PeopleAssignment.self_player`）时只列本人的球，并用“你”称呼。

## 打法风格（`style.py`，不调用模型）

有击球类型时，为每位单打球员（双打为每队）判断打法风格，结果在 `MatchStats.styles`（`PlayerStyle`），特征作为
`<Pn>.style.*` 指标（双打 `A.style.*` / `B.style.*`）出现在统计里，教练可以引用。详细的研究依据、特征和规则见
[`docs/playing-style.md`](../../docs/playing-style.md)。

- **攻防状态**：每一拍按来球判断击球者处于主动 / 中立 / 被动（对手挑球 → 主动，对手杀 / 扑 → 被动），
  统计每种状态下打了什么、符合推荐打法的比例；逐拍建议的 `ShotMoment.phase` 也带上这个状态。
- **打法原型匹配**：单打 8 种（强攻压制、速度抢攻、拉吊突击、控制组织、防守反击、多拍消耗、技术欺骗、节奏变化），
  双打 5 种体系（后场强攻、平抽快挡、网前预判控制、防守转换、混双前封后攻）。每种打法是一组带方向和权重的特征，
  吻合度 0–100；少于 20 拍为“证据不足”，最高分低于 60 为“均衡型”。
- **六种战术套路**（单打）：四角调动、网前逼起球、拉吊突击、防守转攻、重复落点陷阱、节奏破坏，统计次数、所在回合胜率、
  逼对手被动的比例和可回看的视频时间。
- **回球习惯**：对手打某种球后此人最常见的回球（下一拍预测），用来提示对手的预判点。
- **建议**（`PlayerStyle.advice`）：执行问题（主动不进攻、被动顶不住、进攻失误多……）、有效 / 无效的套路、学习对象、
  针对对手打法的克制思路。主动转化低、被动顶不住、进攻失误多也会成为训练重点（`detect_focus`）去检索教学笔记。

## AI 教练（`coach/`）

- Provider 协议：`generate(stats, knowledge=None) -> dict`（符合 `prompt.OUTPUT_SCHEMA`），失败抛 `CoachProviderError`。
  用 `register_provider(name, factory)` 注册新的提供方。
- 默认 `auto`：先用**本地免费模型** `ollama`，不可用时如果设置了 `ANTHROPIC_API_KEY` 再用 `anthropic`，最后用 `offline`；改用了哪个、为什么，写在 `fallback_reason`。
- `ollama`：本机 Ollama 运行的开源模型，默认 `qwen2.5:7b`（中文好、JSON 稳定，约 5 GB 内存），数据不出本机、不花钱。
  安装：`brew install ollama`，`ollama serve`，`ollama pull qwen2.5:7b`。
  环境变量：`RALLYIQ_OLLAMA_URL`（默认 `http://localhost:11434`）、`RALLYIQ_OLLAMA_MODEL`、`RALLYIQ_OLLAMA_CTX`（默认 32768）、
  `RALLYIQ_OLLAMA_TIMEOUT`（秒，默认 900）、`RALLYIQ_OLLAMA_KEEP_ALIVE`（生成后模型在内存里留多久，默认 `5m`）。
  内存紧张可换 `qwen2.5:3b`；内存充足可换 `qwen2.5:14b` 或 `qwen3:8b`。
- `anthropic`（可选）：Claude API，`claude-opus-5-5`，adaptive thinking，结构化 JSON 输出，启用服务端 fallback。
  环境变量：`RALLYIQ_COACH_MODEL`、`RALLYIQ_COACH_EFFORT`（默认 `high`）。
  需要 `pip install -e '.[coach]'` 和 `ANTHROPIC_API_KEY`（或 `ant auth login`）。
- `RALLYIQ_COACH_PROVIDER` 可固定为 `ollama` / `anthropic` / `offline`（失败时仍回退到离线模板）。
- `offline`：规则模板，只说能直接从指标读出的结论。默认提供方不可用时自动回退，并在报告中写明原因。
- **防编造校验**（`grounding.py`）：引用不存在的指标会被去掉，没有有效依据的观点被删除；观点可信度不高于所引用指标中最低的一个；
  文中数字必须能在所引用的数据中找到（允许四舍五入），否则标记“含未核实数字”。

### 基于教学笔记的训练建议（`advice.py`）

1. `detect_focus(stats)` 从数据里找每位球员的问题：失分最后一拍集中在某种击球、自己最后一拍失分多、发球/接发胜率低、长短回合胜率差、覆盖率低、比对手多跑、站位偏后场/前场。每个问题带依据指标和复测指标。
2. `retrieve` 按击球标签、方面标签和笔记的 `related_metrics` 给 `knowledge/coaching/notes` 里的笔记打分，每个问题取 3 篇（先每位教练一篇）。可用 `RALLYIQ_PLAYER_LEVEL=beginner|intermediate|advanced` 偏向对应水平。
3. 问题和精简后的笔记一起发给模型。训练建议填 `focus_id`、练法、理由、`key_points`、引用的 `note_id`；离线模板直接取最匹配笔记的第一个练习。
4. 校验：只能引用这次检索到的笔记，链接和标题从笔记文件取，不由模型生成；复测指标无效时改用问题自带的复测指标。笔记的练习带 `t_s` 时链接直接跳到该时间点。
5. 建议随比赛写入档案，下一场的 `GET /api/profile` 复测结果会带上当初参考的视频。

笔记目录默认是仓库根的 `knowledge/coaching`，可用 `RALLYIQ_KNOWLEDGE_DIR` 指定；没有笔记时照常生成点评，只是不带引用。

## 接口

- `GET /api/videos/{id}/report` → `MatchReport`（统计 + 最近一次点评；数据修正后 `coach_stale=true`）
- `POST /api/videos/{id}/report/coach`（body `{"provider": null}`）→ `CoachReport`，同步生成并缓存
- `GET /api/videos/{id}/report/export?format=md|json`
- `GET /api/coach/providers`
