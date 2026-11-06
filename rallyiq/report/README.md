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
| 得失分 | `Pn.win.won/rate_pct`、`short_rate_pct`、`long_rate_pct`、`serve_rate_pct`、`receive_rate_pct`、`own_last_shot_pct`、`lost_own_last_shot_pct` |

- 移动只统计回合内、非插值的足点；5 帧滑动中值去抖，速度 > 8 m/s 的跳变视为跟踪错误剔除，间隔 > 0.5 s 不连线。
- 站位按“本方半场”换算：深度 = 离自己底线的距离，左右按本人面向球网的视角，换边后仍可比较。
- 球场标定不是 `ok` 时，米制指标降为“待复核”。
- 胜负指标只用已标注胜者的回合（工作台修正面板里设置），没有标注时为“不可判定”。
- **击球类型**：读取 `ShotEvent.shot_type`。击球分类模块接入后填这个字段即可，统计和报告自动出现分布（`SHOT_TYPE_LABELS` 里登记中文名），无需改报告代码。

## AI 教练（`coach/`）

- Provider 协议：`generate(stats) -> dict`（符合 `prompt.OUTPUT_SCHEMA`），失败抛 `CoachProviderError`。
  用 `register_provider(name, factory)` 注册新的提供方。
- `anthropic`（默认）：Claude API，`claude-opus-5-5`，adaptive thinking，结构化 JSON 输出，启用服务端 fallback。
  环境变量：`RALLYIQ_COACH_PROVIDER`、`RALLYIQ_COACH_MODEL`、`RALLYIQ_COACH_EFFORT`（默认 `high`）。
  需要 `pip install -e '.[coach]'` 和 `ANTHROPIC_API_KEY`（或 `ant auth login`）。
- `offline`：规则模板，只说能直接从指标读出的结论。默认提供方不可用时自动回退，并在报告中写明原因。
- **防编造校验**（`grounding.py`）：引用不存在的指标会被去掉，没有有效依据的观点被删除；观点可信度不高于所引用指标中最低的一个；
  文中数字必须能在所引用的数据中找到（允许四舍五入），否则标记“含未核实数字”。

## 接口

- `GET /api/videos/{id}/report` → `MatchReport`（统计 + 最近一次点评；数据修正后 `coach_stale=true`）
- `POST /api/videos/{id}/report/coach`（body `{"provider": null}`）→ `CoachReport`，同步生成并缓存
- `GET /api/videos/{id}/report/export?format=md|json`
- `GET /api/coach/providers`
