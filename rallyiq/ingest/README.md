# ingest：视频输入评估

`GET /api/videos/{id}/quality` 返回 `VideoQuality`：这段视频是什么条件、每类输出能信多少。不阻止分析（"降级而不拒绝"）。

| 检查项 `key` | 来源 | 待复核 / 不可判定的条件 |
| --- | --- | --- |
| `fps` | 探测 | < 25 fps 待复核（30 fps 附说明：击球按区域级精度） |
| `resolution` | 探测 | 短边 < 540 待复核 |
| `vfr` | 探测 | 可变帧率时出现，待复核 |
| `calibration` | 标定 | 可用段覆盖 < 90% 待复核，已标定 < 50% 不可判定 |
| `camera_view` / `camera_height` | 标定外框四角 | 侧面机位、低机位待复核（几何估计，粗分类） |
| `court_in_frame` | 标定外框四角 | 有角在画面外待复核 |
| `camera_motion` | 标定分段 | 每分钟 > 1 次移动 / 切换待复核 |
| `near_player_rate` / `far_player_rate` | 球员（回合内，按抽帧步长） | < 80% 待复核，< 40% 不可判定 |
| `court_choice` | 标定 + 球员 | 标定"可用"但两侧球员可见都 < 15%：疑似选错场地，标定和球场坐标改为不可判定，提示手动点选四角 |
| `shuttle_rate` | 羽毛球（回合内） | < 60% 待复核，< 25% 不可判定 |
| `brightness` / `sharpness` | 抽 12 帧（需要 OpenCV） | 偏暗 / 过曝 / 模糊待复核 |

`capabilities` 把这些检查汇总到五类输出：球场坐标、球员、羽毛球、回合、击球。击球在评估集达到 F1 ≥ 0.90 之前始终是待复核。

结果缓存为 `artifacts/quality.json`，任一输入产物或视频文件变化后自动重算；抽帧统计单独缓存为 `picture_stats.json`。阈值是初版经验值，等评估集按机位 / 帧率分组的结果出来后再调。
