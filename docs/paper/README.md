# RallyIQ 技术论文

《RallyIQ：本地化羽毛球视频分析系统的算法与架构》（2026-11-07，中文，19 页）

- [rallyiq-paper.pdf](rallyiq-paper.pdf)：全文
- [architecture.png](architecture.png)：系统架构图（感知层 → 理解层 → 分析与输出，人工修正闭环）
- [hit-detection.png](hit-detection.png)：击球检测流程图
- 第 3.11 节：8 个核心算法的伪代码（标定、羽毛球跟踪、击球检测、交替约束动态规划、击球方补全、冠军制微调、打法原型匹配、脚步）

内容：系统架构；球场标定、TrackNet 式羽毛球检测与 `shuttle_loop` 持续微调、RTMO 姿态与身份追踪、
回合与击球检测及击球方归属、HistGB 击球类型与多层描述、打法风格、脚步、赛后报告与 AI 教练；
ShuttleSet / TrackNet 上的评测数字（按比赛分组、整场留出）；局限与后续计划。

数字取自 `runs/` 实测记录和各模块 README（主干 2382d53）。多层描述、打法风格、脚步三节描述的是尚未合并的开发分支。
