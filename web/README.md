# RallyIQ web 工作台

单页 React 应用（Vite + TypeScript），对接 `rallyiq/api`（FastAPI）。功能：

- 上传视频（显示上传进度），视频列表和分析任务状态（逐阶段进度轮询）
- 回放：原画 / AI 标注切换；双色骨架、羽毛球轨迹（实线 = 检测，虚线半透明 = 插值）、球场线、检测框，各层可开关、调透明度
- 时间轴：回合色块、击球刻度（按击球者着色，待复核标 `!`），点击跳转，可放大到当前回合；慢放、逐帧
- 左栏：视频与标定状态、跑位热力图（P1/P2，整段或当前回合，标定可靠时才显示跑动米数）、手动点选球场四角重新标定
- 右栏：球场小地图（带 2 秒轨迹）、当前回合与击球列表、纠错、修正记录、导出
- 击球类型：击球列表和时间轴（放大回合时）显示类型，虚线框 + `?` 表示待复核；选中一拍可看置信度、候选、待复核原因和判断依据，并一键改类型或确认；左栏按球员统计类型分布（整场 / 当前回合）和"最近一拍"
- 纠错（全部追加写入，原始预测保留）：增 / 删 / 移击球、改击球者、改击球类型、确认；改回合起止、删除 / 新建 / 确认回合、设得分方；从某帧起交换 P1/P2。默认勾选"记为难例"（`hard_example`）
- 导出 JSON（生效事件 + 原始预测 + 修正）和逐拍 CSV

快捷键：`空格` 播放/暂停，`←` `→` 逐帧，`J` `K` 上/下一拍，`[` `]` 上/下回合。

## 运行

```bash
# 后端（仓库根目录）
pip install -e '.[dev]'
RALLYIQ_DEMO=1 uvicorn rallyiq.api.app:app --port 8000

# 前端
cd web
npm install
npm run dev          # http://localhost:5173 ，/api 代理到 :8000（RALLYIQ_API 可改）
```

不启动后端也能看界面：打开 `http://localhost:5173/?mock`（或 `VITE_USE_MOCK=1 npm run dev`），使用前端生成的合成数据，修正存在浏览器 localStorage。

```bash
npm test             # vitest：纠错逻辑与后端 contracts/examples 一致性、单应性、热力图、CSV
npm run build        # 类型检查 + 生产构建到 web/dist
```

## 代码结构

| 文件 | 内容 |
| --- | --- |
| `src/lib/types.ts` | 与 `rallyiq/contracts/models.py` 对应的接口类型 |
| `src/lib/api.ts` | 真实 API 客户端 + 离线 mock 实现 |
| `src/lib/strokes.ts` | 击球类型名称、颜色、原因码翻译、按球员统计、`mergeStrokes` |
| `src/lib/corrections.ts` | `rallyiq/api/corrections.py` 的 TS 移植（mock 模式用），有一致性测试 |
| `src/lib/overlay.ts` | Canvas 叠加层绘制 |
| `src/lib/clock.ts` | 统一播放时钟（有视频用 `<video>`，无视频用虚拟时钟） |
| `src/lib/mock.ts` | 确定性合成数据（含漏检、误检、击球者错误，便于演示纠错） |
| `src/pages/Workbench.tsx` | 三栏工作台 |
