# RallyIQ

**个人 AI 羽毛球训练与比赛分析系统：学习正确打法，读懂比赛，预测对手战术。**

> 状态：MVP 开发中。后端骨架、任务队列、共享数据契约、球员/羽毛球感知、回合与击球候选、击球类型分类和网页工作台已就绪。本 README 是研发蓝图的摘要。

## 本地部署

RallyIQ 在你自己的电脑上运行：视频、分析结果和个人档案都只存在本机（默认 `./data`），不需要任何云服务或账号。

### 系统要求

| 项目 | 要求 |
| --- | --- |
| 系统 | macOS 13+（推荐 Apple Silicon，PyTorch 走 MPS、姿态模型走 CoreML）；Linux x86_64（有 NVIDIA 显卡时自动用 CUDA，没有也能用 CPU 跑，只是慢）。Windows 请用 WSL2（未专门测试） |
| 内存 | 16 GB 以上；如果同时用本地 AI 教练（Qwen 7B），再多留约 5 GB |
| 磁盘 | 依赖和模型约 3–5 GB，另加视频本身（手机 1080p 约 6 GB/小时） |
| 软件 | Python ≥ 3.11、FFmpeg（含 ffprobe）、Node.js ≥ 20（只用来构建网页界面）、git |

安装前置软件：

```bash
# macOS（Homebrew）
brew install python@3.12 ffmpeg node git
# Ubuntu / Debian
sudo apt install python3 python3-venv ffmpeg git
# Node.js：发行版自带的往往太旧，建议用 nvm（https://github.com/nvm-sh/nvm）：nvm install 22
```

### 一键安装并启动

```bash
git clone https://github.com/yangzh0728-lgtm/RallyIQ.git
cd RallyIQ
bash scripts/setup_local.sh     # 检查依赖 → 建 .venv → 下载模型权重 → 构建网页（重复运行会跳过已完成的步骤）
bash scripts/start_local.sh     # 打开 http://localhost:8000
```

在网页上拖入一段比赛或训练视频，上传后自动开始分析，页面会显示各阶段进度；完成后点进视频即可回放、看骨架和球轨迹、逐拍修正、导出。按 `Ctrl+C` 停止服务。

### 分步安装（想自己控制每一步时）

```bash
# 1. Python 环境和分析模型的依赖（PyTorch、ONNX Runtime、OpenCV）
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[calibration,players,shuttle]'

# 2. 模型权重 → weights/pose.onnx（RTMO 姿态，Apache-2.0）和 weights/shuttle/heatmap.pt（羽毛球检测，MIT）
bash scripts/fetch_weights.sh            # POSE_VARIANT=rtmo-m 换更大的姿态模型（远端球员更准、更慢）

# 3. 网页界面
(cd web && npm ci && npm run build)

# 4. 启动（API、分析队列和网页在同一个进程里）
RALLYIQ_WEB_DIR=web/dist uvicorn rallyiq.api.app:app --port 8000
```

- 模型放在 `weights/` 下会被自动找到，不用设环境变量；放在别处时用 `RALLYIQ_POSE_MODEL`、`RALLYIQ_SHUTTLE_WEIGHTS` 指定。
- 羽毛球权重托管在 Google Drive。下载不了时，手动打开 `scripts/fetch_shuttle_weights.py` 里的 `DRIVE_URL` 下载 zip，
  再运行 `python scripts/fetch_shuttle_weights.py --zip 下载的文件.zip`。
- 缺少某个模型时不会报错：对应阶段输出空结果（界面上没有骨架或球轨迹），其余阶段照常运行，日志里会提示缺哪个模型。
- 开发模式：`uvicorn rallyiq.api.app:app --reload --port 8000`，另开终端 `cd web && npm run dev`，打开 http://localhost:5173
  （`/api` 自动转发到 8000）。不装任何模型也能先看界面：`RALLYIQ_DEMO=1` 让各阶段输出合成数据，或打开 `?mock` 演示模式。

### 不开网页，直接分析一段视频

```bash
source .venv/bin/activate
python -m rallyiq.pipeline analyze match.mp4    # 打印各阶段状态和结果目录 data/videos/<video_id>/artifacts/
```

分析过的视频也会出现在网页的视频列表里。结果是 JSON（球场标定、球员轨迹、羽毛球轨迹、回合与击球、击球类型、叠加层），
格式见 `contracts/README.md`；网页上可以导出 JSON / CSV、赛后报告（Markdown）和回合片段（MP4）。

### 可选：本地 AI 教练点评（Ollama + Qwen）

赛后报告里的"教练点评"默认用本机运行的开源大模型，免费、数据不出本机：

```bash
# macOS
brew install ollama
# Linux
curl -fsSL https://ollama.com/install.sh | sh

ollama serve &                # 已作为系统服务运行时可省略
ollama pull qwen2.5:7b        # 约 4.7 GB；内存充足可换 qwen2.5:14b，并设置 RALLYIQ_OLLAMA_MODEL=qwen2.5:14b
```

RallyIQ 会自动连接 `http://localhost:11434`（`RALLYIQ_OLLAMA_URL` 可改）。没有装 Ollama 时，报告改用离线规则模板并注明原因；
也可以改用 Claude API：`pip install -e '.[coach]'` 并设置 `ANTHROPIC_API_KEY`（按调用付费）。详见 `rallyiq/report/README.md`。

### 常见问题

- **怎么确认用上了 GPU？** `setup_local.sh` 结束前会打印 `torch … -> mps / cuda / cpu` 和 ONNX Runtime 的可用后端。
  Apple Silicon 第一次分析时 CoreML 要编译姿态模型，会多等一两分钟，之后有缓存（`~/.cache/rallyiq/coreml`）。
- **NVIDIA 显卡上的姿态模型**：默认的 `onnxruntime` 只用 CPU；需要时 `pip install onnxruntime-gpu`（先 `pip uninstall onnxruntime`）。
- **数据在哪、怎么删？** 全部在 `RALLYIQ_DATA`（默认 `./data`）：`rallyiq.db`（SQLite）和 `videos/<id>/`。网页上每个视频都有"删除"按钮，
  会一起删掉视频、分析结果和修正记录。备份时复制整个目录即可。
- **换端口 / 让同一局域网的手机访问**：`PORT=8080 HOST=0.0.0.0 bash scripts/start_local.sh`。默认没有登录，只在可信网络里这样做；
  需要账号时设置 `RALLYIQ_AUTH=on RALLYIQ_INVITE_CODE=自定义邀请码`（见 `rallyiq/api/auth.py`）。
- **更新到新版本**：`git pull && bash scripts/setup_local.sh`，已有数据会保留，数据库表结构自动升级。
- **在服务器或多台机器上部署**（Docker、PostgreSQL、S3 存储、独立 GPU worker）是可选的进阶用法，见 [`docs/deploy.md`](docs/deploy.md)。

## 项目定位

RallyIQ 把视觉感知、人体与球拍运动学、羽毛球轨迹、击球时序、战术推断和训练复测连成一个闭环，并持续积累成一份**可更新的个人打法档案**。目标是真正提高使用者的比赛决策和竞技表现，而不只是生成好看的视频标注。

### 核心原则

- **只服务一个人**：单用户私有系统，不做多租户、社交、订阅。
- **正确打法是条件化的**：取决于球员能力、击球情境和比赛目标，不用单一职业动作模板评判所有人。
- **分级表达结论**：区分视频事实、统计推断和战术建议，每条都附证据和不确定性；无法判断时输出"证据不足"。
- **不污染真实数据**：未经确认的模型预测和 LLM 总结不能写成真值；人工修改保留历史版本。
- **不过度承诺**：普通 30fps 单摄像头适合区域级分析，不默认输出厘米级落点、精确关节角速度或伤病判断。每项任务按"可用 / 待复核 / 不可判定"输出。

## 功能范围

| 优先级 | 内容 |
| --- | --- |
| **P0** | 单打分析；球场标定；球员与羽毛球追踪；击球帧、击球类型、目标区域与回合重建；个人历史记录；对手报告 |
| **P1** | 姿态与球拍检测；技术动作分段；对手条件概率；自动训练推荐与复测 |
| **P2** | 多视角三维轨迹；双打（四人身份与搭档关系）；音视频联合识别；模拟比赛；实时推理 |

## 识别能力优先，而不是限制拍法

RallyIQ 不要求使用者按固定的机位、高度、帧率或流程录像。手机随手拍的视频也应该能分析，系统的工作是**提高识别能力去适应视频**，而不是让人去适应系统。

- **自动评估输入**：识别机位（后场正中、偏斜、侧面、高位、低位、转播画面）、帧率、分辨率、抖动、光照和遮挡，据此选择处理方式，并说明每项结论在这段视频上的可信程度。
- **鲁棒的球场标定**：自动检测场线与交点，用额外交点独立核验，识别镜像和左右翻转；检测到镜头移动或切换时自动重新标定。自动失败时才请人点选四角。
- **远端与小目标**：针对远端球员、远端小球、高速模糊和背景干扰专门评估和改进，而不是只在理想画面上调参。
- **低帧率与丢球**：30 fps、运动模糊或球出画时，保留观测缺口并单独标记补点，不伪造轨迹。
- **身份与换边**：依靠球场半区和轨迹连续性维护两人身份，自动检测换边；不确定时请人确认一次。
- **降级而不拒绝**：画质不足时仍输出能可靠给出的部分（例如回合和跑位），其他项标为"待复核"或"不可判定"。
- **按条件分组评估**：评估集刻意覆盖不同机位、帧率、场馆和光照，按条件分别报告精度，找出最弱的场景优先改进。

## 一次分析的流程

1. 上传视频，并选择问题：分析我的训练 / 分析比赛 / 分析某个对手 / 与历史比较。
2. 检查视频质量：帧率、分辨率、镜头移动、场地可见度、视角。
3. 自动标定球场并运行感知模型，抽取击球事件和姿态；低置信结果进入人工复核列表。
4. 针对问题输出统计、结论和对应的视频时间戳。
5. 把已验证的结果写入个人长期档案，记录新增、冲突和趋势变化。
6. 重新排列训练优先级，生成训练单和复测指标；下次上传时自动对比。

## 系统组成

### 感知层（看见）
- **视频与球场**：机位识别；球场线检测，按标准 13.40 × 6.10 m 球场估计单应性；标定失败可识别，并允许手动点选四角。
- **羽毛球**：多检测器对比，时序轨迹模型，物理先验；严格区分"预计目标区"和"真实落点"。
- **球员与球拍**：球员身份跟踪，人体关键点及置信度，球拍检测，技术动作分段（准备、引拍、击球、随挥、恢复）。

### 事件层（还原）
- 标准化的 `ShotEvent` 和 `RallyEvent`：字段允许为空，不填伪造的默认值。
- 回合状态机：发球 → 击球 → 过网 → 落地 / 失误 → 回合结束；比分归属不确定时标为"待确认"。
- 统一使用原视频毫秒时间与球场世界坐标（米）。

### 知识层（理解）
- "正确打法"分四层：规则、生物力学、情境战术、结果验证。
- 专家示范、教练纠错案例、本人成功与失败动作三个案例库；每条知识注明出处。
- 比赛阅读：主动 / 被动状态、站位、节奏和战术意图，而不只是识别击球名称。

### 策略层（行动）
- 对手画像：条件化击球分布、下一拍预测、弱点假设；报告样本量、可信度、视频证据和反例。
- 个性化训练：根据个人能力生成训练单，训练后上传新录像，在同条件下复测比较。

### 个人记忆
- 采用只追加的分析事件日志（`AnalysisEvent`），个人画像（`PlayerProfile`）由日志重新计算，新上传不会覆盖旧证据。
- 档案内容包括技能、移动、比赛、训练和对手，每项都带确认状态、模型版本和原视频引用。

## 技术架构（初定）

本地优先，可选私有 GPU 服务。

| 部分 | 选型 |
| --- | --- |
| 视觉与推理 | Python、PyTorch、OpenCV、FFmpeg |
| API | FastAPI |
| 结构化数据 | SQLite 起步，可迁移到 PostgreSQL |
| 视频与模型权重 | 本地文件系统或私有对象存储 |
| 前端工作台 | Next.js / React，视频叠加层用 Canvas / WebGL 绘制 |

### 模块划分

```
ingest/       上传、视频解析、切片、机位与质量评估
perception/   球场、球员、球拍、骨架、羽毛球检测与追踪
events/       击球、回合、落地与比分事件
analytics/    动作、移动、空间、序列与比赛分析
memory/       个人档案事件存储、对手档案、训练历史
planner/      战术概率、训练推荐、报告与证据关联
review/       人工复核、修正与标注回流
evaluation/   模型基准、可复现实验、回归测试
```

## 代码结构与本地运行（MVP）

```
rallyiq/contracts/              共享数据契约（pydantic）与球场几何；见 contracts/README.md
rallyiq/perception/calibration/ 球场标定阶段
rallyiq/perception/players/     球员姿态与跟踪阶段
rallyiq/perception/shuttle/     羽毛球轨迹阶段
rallyiq/events/                 回合与击球候选阶段
rallyiq/strokes/                击球类型分类阶段（12 类，规则 v1；可选接入 BST 模型）
rallyiq/ingest/                 视频输入评估（机位、帧率、画质、各项输出可信度）
rallyiq/memory/                 个人长期档案：只追加的分析事件日志、球员画像与训练复测、对手习惯报告
rallyiq/review/                 难例收集：人工修正按视频条件分组，找出最弱场景
rallyiq/pipeline/               阶段注册、任务执行、队列 worker、视频探测、叠加层合成
rallyiq/report/                 赛后统计与 AI 教练点评（可插拔 LLM，默认本地 Ollama + Qwen）
rallyiq/api/                    FastAPI 后端（上传、任务、结果、修正、报告、导出）
rallyiq/store.py                数据库（db.py：SQLite / PostgreSQL）+ 文件存储（blobs.py：本地 / S3）
rallyiq/api/auth.py             账号与会话（默认关闭，线上 RALLYIQ_AUTH=on）
scripts/setup_local.sh 等        本地安装与启动；Dockerfile、docker/、deploy/ 为可选的服务器部署
contracts/schemas, examples     由契约自动生成的 JSON Schema 和示例
web/                            网页回放与修正界面
```

开发与测试（安装和运行见上方"本地部署"）：

```bash
pip install -e '.[dev]'                       # 需要 ffmpeg/ffprobe
RALLYIQ_DEMO=1 uvicorn rallyiq.api.app:app --port 8000   # 未接入模型的阶段输出合成数据
python -m rallyiq.pipeline analyze clip.mp4 --demo       # 离线跑一遍流水线
pytest
```

主要接口（均在 `/api` 下）：`POST /videos` 上传并自动排队分析；`GET /jobs/{id}` 查看各阶段进度；`GET /videos/{id}/file` 视频（支持 Range）；`GET /videos/{id}/overlay` 叠加层；`GET /videos/{id}/events` 应用修正后的回合与击球；`POST /videos/{id}/corrections` 追加修正（含 `set_shot_type` 改击球类型）；`PUT /videos/{id}/calibration/manual` 点选四角；`PUT /videos/{id}/match_format` 指定单打 / 双打 / 自动并重跑；`GET /videos/{id}/clip?rally_id=…`（或 `start_ms`/`end_ms`）导出回合片段 MP4；`GET /videos/{id}/quality` 视频条件与各项输出可信度；`PUT /videos/{id}/people` 标注谁是我、对手是谁；`POST /videos/{id}/memory/commit` 把确认过的比赛写入档案（`/memory/withdraw` 撤下，均为追加）；`GET /profile`（`?person=`）个人画像、趋势与训练复测；`GET /opponents/{name}` 对手习惯；`GET /people`；`GET /memory/log`；`GET /hard_examples`（`?format=jsonl`）与 `GET /hard_examples/summary` 难例及按条件分组的错误率（命令行：`python -m rallyiq.review export --out hard_examples/`）；`GET /videos/{id}/heatmap`（可按 `player_id` 或双打 `team`）；`GET /videos/{id}/footwork` 脚步分析（触地、步伐、启动步、弓步、起跳、并步/交叉步、回位，见 `rallyiq/footwork/README.md`）；`GET /videos/{id}/export?format=json|csv`；`GET /videos/{id}/report` 赛后统计 + 最近一次教练点评；`POST /videos/{id}/report/coach` 生成 AI 点评；`GET /videos/{id}/report/export?format=md|json`。完整文档见运行后的 `/docs`。

AI 教练默认用本机免费的开源模型（Ollama + `qwen2.5:7b`：`brew install ollama && ollama serve && ollama pull qwen2.5:7b`），数据不出本机；设置了 `ANTHROPIC_API_KEY` 时可改用 Claude；都不可用时自动改用离线规则模板并在报告中注明。详见 `rallyiq/report/README.md`。

## 界面设计

深色三栏"运动视觉工作台"，桌面优先（1440×900 以上）：

- **中央视频区（60%–70%）**：原画与 AI 标注层一键切换；双色球员骨架、黄色羽毛球轨迹（观测为实心、插值为半透明、预测为虚线）、球拍轨迹、球场线与透视网格。
- **左栏**：热力图（单回合 / 单局 / 整场）、跑动距离、覆盖区、回位时间，并标明样本量和可信度。
- **右栏**：双方跑动轨迹、球场小地图、最新击球类型和下一拍候选概率。
- **底部时间轴**：回合切片、关键击球点、置信度警示、慢放、逐帧、导出片段。
- **战术重建视图**：用简化的球场和骨架重新表达比赛，便于阅读站位和节奏。
- 所有叠加层可单独开关、调整透明度；只有在空间标定可靠时才显示米和 m/s；同一球员全程使用同一颜色；支持中英文。

## 首版 MVP：2 周快速推出（单用户、离线分析）

**原则**：先用最小范围跑通"上传 → 识别 → 回放 → 修正"的完整闭环并尽快推出，之后每周按真实使用中最常见的识别错误迭代。不追求首版精度达标，但每个输出都标明是自动结果还是待复核。

**交付目标**：上传一段手机随手拍的单打视频，几分钟到几十分钟后在网页上看到：两人骨架和跑位、羽毛球轨迹、自动切出的回合和击球候选，能逐拍跳转、修正误判，并导出带时间戳的结果。

### 首版做什么

| 部分 | 方案 |
| --- | --- |
| 球场 | 自动场线检测 + 单应性；失败时点选四角兜底 |
| 球员 | 现成轻量姿态模型；足点映射到球场；用半区和轨迹连续性区分两人 |
| 羽毛球 | 现成时序轨迹检测模型直接推理；原始检测与补点分开保存 |
| 回合与击球 | 规则法：球路转折 + 腕部邻近提出击球候选；球长时间消失或静止判定回合结束 |
| 界面 | 一个网页：视频 + Canvas 叠加层（骨架、球轨迹）+ 时间轴（回合、击球点）+ 跑位热力图 |
| 修正与导出 | 新增 / 删除击球、修改击球者；导出 JSON / CSV，保留原始预测 |
| 存储 | Python 离线任务 + SQLite + 本地视频文件 |

### 首版不做（推出后再加）

个人长期档案与对手报告、抽象 / 双屏视图、球拍检测、模型训练与微调。

> 双打已提前接入：自动识别单 / 双打，四人稳定 ID 与 A/B 队伍（A = P1+P3，B = P2+P4），击球按队伍交替归属，双打边界线，热力图可按球员或队伍查看；识别不对时可在网页上强制指定。

### 排期（约 30 小时）

| 时间 | 内容 | 完成标准 |
| --- | --- | --- |
| 第 1–2 天 | 跑通环境；球与姿态模型在本机对一段视频出结果，测处理速度 | 一段 60 秒视频得到逐帧球和骨架数据 |
| 第 3–4 天 | 球场自动标定 + 兜底点选；足点映射；两人 ID | 叠加到原片上的球场线和足点位置正确 |
| 第 5–6 天 | 回合切分与击球候选规则；导出事件 | 一段完整视频输出回合和击球列表 |
| 第 7–9 天 | 网页回放：叠加层、时间轴、逐拍跳转、热力图 | 能在浏览器里逐拍查看 |
| 第 10 天 | 修正与导出；用 2–3 段不同条件拍的视频走完整流程 | **推出首版** |

### 推出后：以提升识别能力为主线迭代

- 每次使用时的人工修正自动记录为难例，积累成评估集，按机位、帧率、场馆和光照分组。（已实现：`rallyiq/review`，首页“难例与薄弱场景”）
- 每周挑出最弱的场景（例如远端小球、侧面机位、30 fps）针对性改进，并用同一评估集对比前后。
- 当击球检测 F1 ≥ 0.90、可见球召回 ≥ 0.90（同一球员 ±3 帧匹配；1080p 下 ≤ 12 像素算命中）时，该功能才去掉"待复核"标记。
- 击球类型分类已提前接入（规则 v1，见 `rallyiq/strokes/README.md`），所有类型默认待复核，直到真实标注验证；之后依次加入个人档案、对手报告和 AI Coach。

### 算力与预算

先用现有电脑（Apple M4 Pro，48 GB 统一内存）运行；第 1 天实测处理速度。默认新增现金支出为 0。

### 依赖准入

所有代码、模型权重和数据集在引入前登记许可证和版本；许可不明确的不引入。首版只使用自己拍摄的数据。

## 长期里程碑

| 阶段 | 内容 | 验收标准 |
| --- | --- | --- |
| M0–M3 | 环境、球场层、感知、击球与回合事件 | 首版见上方 MVP，推出后持续提高识别精度 |
| M4 | 正确打法知识库 | 对可观察的技术问题给出有证据的反馈（需独立复核，如教练盲评） |
| M5 | 个人长期记忆 | 两次以上上传可追溯地更新画像，不覆盖原始事实 |
| M6 | 对手战术学习 | 报告附样本量、可信度、视频证据和反例 |
| M7 | 训练与比赛策略 | 训练建议匹配个人能力，改善可用录像验证 |
| M8 | 跨机位泛化与高级模型 | 在未见过的机位和场馆上达到单独制定的门槛；多机位同步与三维轨迹 |
| M9 | 持续研发 | 主动学习挑选难例标注；每次更新有基准对照、版本记录和回滚能力 |

## 第一个完整闭环

1. 用手机录一场单打比赛并上传，不限定机位。
2. 自动完成球场标定、球员识别、球轨迹和回合 / 击球事件，生成证据时间轴。
3. 输出自己的技术问题和对手前 3 项可检验的习惯（允许"证据不足"）。
4. 报告中每条结论都能点击跳到原视频。
5. 生成下一次训练的 2–3 项任务；训练后上传新录像，在同条件下对比。
6. 更新个人画像、弱点可信度、训练记录和下一步计划。

## 数据与隐私

- 只分析本人视频，以及合法取得的对手录像。
- 本地存储，支持磁盘加密、备份、导出和删除。
- 不输出无法从画面确定的伤病、病理或情绪判断。
