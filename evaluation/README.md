# evaluation/ — BWF 比赛评测集与识别基准

用 BWF TV 官方 YouTube 频道（@BWF）的比赛录像，按条件分组测量各识别模块的精度，
并为微调准备带标注的数据。

## ⚠️ 版权与使用范围

- 这些视频归 BWF 所有；YouTube 服务条款限制下载。**仓库只保存 URL、时间段和标注，
  不保存任何视频、帧图像或裁剪图。** `.gitignore` 已排除常见视频格式。
- 仅用于个人内部研究和识别精度评测，不再分发、不公开发布由此训练出的模型权重、不商用。
- 是否下载由你自行决定。抽片工具默认只处理你已放在本地的视频；`--download` 选项
  会调用 yt-dlp 只下载清单里的 1–2 分钟区间。
- 这与主 README "首版只使用自己拍摄的数据" 不同：BWF 片段只做**评测与研究**，
  不进入产品默认流程；主 README 的依赖准入表应登记这一数据来源。

## 片段清单 `clips.yaml`

19 个片段，覆盖：

| 条件 | 覆盖 |
|---|---|
| 项目 | 男单、女单、男双、女双、混双 |
| 场馆 | 伯明翰、巴黎、东京、雅加达、欧登塞、萨尔布吕肯、丽水、杭州 |
| 机位 | 主转播机位；副场固定机位直播流（ko26_court2/3）；剪辑集锦（频繁切镜头） |
| 划分 | `eval`（只评测）12 个，`train`（可用于微调）7 个，同一比赛不跨划分 |

时间段是按经验估计的（`segment_verified: false`）：抽片后先快速看一眼，若是暂停、
颁奖或观众镜头就调整 `start_s` 并改为 `true`。

## 流程

```bash
pip install -r evaluation/requirements.txt

# 1. 抽片：原视频放到 ~/rallyiq-data/bwf/raw/<video_id>.mp4，或加 --download
python -m evaluation.tools.extract_clips
#    → ~/rallyiq-data/bwf/clips/<clip_id>.mp4 + .meta.json（起点毫秒、fps、分辨率）

# 2. 用流水线处理片段，再把产物转换成评测格式（自动加上片段在原视频中的起点）
python -m evaluation.from_contracts data/videos/<video_id>/artifacts \
    --clip-meta ~/rallyiq-data/bwf/clips/ae26_ms_f.meta.json --out preds/

# 3. 标注：用预测打底，人工修正最快
python -m evaluation.tools.label_tool ae26_ms_f --prefill preds/ae26_ms_f.json
#    按 r 标为"已复核"后才计入正式基准

# 4. 基准
python -m evaluation.benchmark --pred preds --markdown out/bench.md --json out/bench.json
```

## 标注内容与工作量建议

| 模块 | 标注 | 每片段建议 |
|---|---|---|
| 球场标定 | 外边界四角（远左、远右、近右、近左）；特写/回放帧标"不可见" | 每次镜头切换后 1 帧，约 10–20 帧 |
| 羽毛球 | 每帧球心位置或"不可见" | 2–3 个完整回合逐帧（约 500–1000 帧） |
| 击球 | 击球帧 + 近端/远端 | 整段 |
| 回合 | 发球到落地/失误的起止 | 整段 |

所有时间都是**原视频毫秒时间**，坐标是**源视频像素**，评测时换算到 1080p。

## 指标（阈值取自主 README 的验收标准）

| 模块 | 指标 |
|---|---|
| 球场 | 标定成功率（平均角点误差 ≤10px@1080p）、角点误差、误标定率（球场不可见时仍输出标定） |
| 羽毛球 | 精确率 / 召回 / F1（≤12px@1080p 命中），并单列**远场球召回**（球的画面位置投影到网后） |
| 击球 | F1（同一侧 ±3 帧内匹配）、不分击球方的 F1、平均时间偏差（帧） |
| 回合 | F1（时间 IoU ≥0.5）、起止时间误差 |

报告按项目、机位、来源、场馆分组，并列出最弱的 3 个条件，作为下一轮改进的优先级。
没有预测文件的片段按全部漏检计分。门槛：击球 F1 ≥ 0.90、可见球召回 ≥ 0.90。

## 测试

```bash
python -m pytest evaluation/tests
```
