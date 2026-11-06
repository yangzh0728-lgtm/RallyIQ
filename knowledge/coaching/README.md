# 教练知识库：球星教学视频笔记

这里存放从职业球员公开教学视频整理出的**逐期笔记**，以后给 AI 教练（`rallyiq/report/coach`）做检索参考：比赛报告发现某项弱点时，可以按击球类型或训练方面找到相应的要点、常见错误和练习方法。

- `INDEX.md`：所有笔记的一览表，直接列出每期教的具体击球和具体步法（由 `scripts/build_coaching_index.py` 生成）。
- `readable/<球员>/<视频ID>.md`：每篇笔记的可读版，同样由脚本从 JSON 生成。
- `catalog.json`：每位球员已发现的教学视频，以及每期的状态（`noted` 已写笔记 / `skipped` 跳过及原因 / `pending` 待处理）。
- `taxonomy.json`：笔记使用的标签词表。`strokes` 与 ShuttleSet 的击球类型对应，方便以后接上击球分类；`aspects` 是步法、握拍、体能等非击球主题。
- `notes/<球员>/<视频ID>.json`：一期视频一个文件。

## 来源与版权

每条笔记都附原视频链接，内容是根据视频字幕（YouTube 自动字幕或上传字幕）和简介**用中文改写的学习笔记**。仓库里不保存字幕全文，也不保存视频文件。教练模型引用时应注明出处（球员 + 视频链接），不要冒充球员本人的原话。

## 笔记字段

| 字段 | 含义 |
|---|---|
| `id` | `<球员>-<平台>-<视频ID>` |
| `coach` / `platform` / `channel` / `url` / `title` / `series` / `published` / `duration_s` | 来源信息 |
| `language` / `transcript_source` | 原视频语言、字幕来源（见 taxonomy） |
| `level` | `beginner` / `intermediate` / `advanced` / `all` |
| `topic` | 一句话主题 |
| `strokes` / `aspects` | 标签，取值必须在 `taxonomy.json` 中 |
| `summary` | 3–5 句概述 |
| `key_points` | 技术要点 |
| `common_mistakes` | 常见错误 |
| `drills` | 练习方法，`{name, how}` |
| `coach_cues` | 简短口令，适合直接放进点评 |
| `shots` | 具体击球：`name_zh` 中文名、`stroke` 类型、`hand` 正手/反手/头顶区、`zone` 前/中/后场、`direction` 直线/斜线 |
| `footwork` | 具体步法（启动步、上网步序、退后场转体并步等） |
| `technique_breakdown` | 动作分解：握拍 / 准备 / 引拍 / 击球 / 随挥 等阶段 |
| `when_to_use` | 使用时机和战术情境 |
| `corrections` | 纠错实例：`problem` 问题、`cue` 他给的提示、`result` 效果 |
| `progression` | 进阶练习顺序 |
| `self_check` | 自查清单 |
| `detailed_description` | 详细描述，按视频顺序讲清楚这一期教了什么、怎么做、为什么，至少 400 字 |
| `related_metrics` | 与这期内容相关、可用来复测的 RallyIQ 指标 key（`{pid}` 为球员前缀） |
| `content_caveats` | 字幕局限、个人习惯不宜照搬等提醒 |

`tests/report/test_coach_knowledge.py` 会检查字段齐全、标签合法、目录与笔记一致。

## 进度

| 球员 | 平台 | 状态 |
|---|---|---|
| 安赛龙 Viktor Axelsen | YouTube | 教学类播放列表 33 期：24 期已写笔记，9 期跳过（无字幕、1 分钟花式球展示、营养补剂/器材内容） |
| 郑思维 | B 站 | 《羽球思维》共 132 期，已用 B 站 AI 字幕写好 101 期，其余进行中；空间里 22 条 Vlog、广告、宣传、点评视频跳过；付费系统课不收录；抖音是同一系列，不重复收录 |

## 新增笔记

1. 在 `notes/<球员>/` 下新建 `<视频ID>.json`，字段同上。
2. 在 `catalog.json` 对应条目把 `status` 改成 `noted` 并填 `note` 路径。
3. 运行 `python scripts/build_coaching_index.py` 更新索引和可读版，再跑 `pytest tests/report/test_coach_knowledge.py`。
