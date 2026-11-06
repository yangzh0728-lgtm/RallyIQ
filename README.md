# RallyIQ · 羽毛球 AI 分析网站

上传一段羽毛球比赛视频，自动生成跑位轨迹、热力图、跑动距离/速度和回合统计。

## 目标

- **第一阶段（原型）**：基于开源项目 [Good-Badminton](https://github.com/yo-WASSUP/Good-Badminton)（Apache 2.0）搭建网站：上传视频 → 后台 GPU 分析 → 在线查看标注视频、热力图和数据。
- **第二阶段（产品化）**：任务排队、视频存储、用户账号、更好的前端界面。
- **第三阶段（深度分析）**：
  - 用 TrackNet 类模型提升羽毛球追踪精度
  - 击球时刻识别、动作分类（高远球、杀球、网前等），需要标注数据训练
  - 支持双打

## 计划架构

| 部分 | 方案（初稿） |
| --- | --- |
| 前端 | Web 页面：上传视频、查看进度和结果 |
| 后端 API | Python（FastAPI），接收视频、创建分析任务 |
| 分析 worker | GPU 机器上运行 Good-Badminton 分析流程 |
| 存储 | 对象存储保存原视频和结果文件 |

## 现有基础（来自 Good-Badminton）

- 姿态检测（RTMPose / RTMO / YOLO-Pose），以脚踝中点定位球员
- YOLO 羽毛球检测
- 球场四角标定 + 自动检测，映射到标准 6.1 × 13.4 m 球场
- 输出：标注视频、`detections.jsonl` 逐帧数据、热力图、散点图

已知限制：仅支持单打、不识别击球和动作、需要固定机位。

## 致谢与许可

本项目的分析流程基于 [yo-WASSUP/Good-Badminton](https://github.com/yo-WASSUP/Good-Badminton)，遵循 Apache License 2.0，并保留原作者署名。
