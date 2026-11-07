# RallyIQ 服务器部署（可选，进阶）

> RallyIQ 的主要用法是**在自己电脑上本地运行**，见 README 的"本地部署"。本文只适合想把它放到服务器上、
> 给多人用或者把分析放到独立 GPU 机器上的人。下面以 Fly.io + Neon + Cloudflare R2 + Modal 为例，
> 其中任何一项都可以换成自己的服务器（`docker compose` 一套即可）。

目标：把 RallyIQ 作为一个在线网站运行，手机或电脑打开网址即可上传视频、查看分析结果；
GPU 只在有分析任务时开机计费，没人用时整站接近零成本。

> 价格为 2025–2026 年各家官网公开价的概数，实际付费前请以官网为准。GPU 处理速度尚未在云端实测，
> 下面的耗时和费用是估算，首次上线后用 `scripts/smoke_online.py` 和真实视频实测再更新本文。

## 1. 架构

```
浏览器 ──HTTPS──▶ Fly.io：web + API 容器（Dockerfile，CPU，无流量自动停机）
   │                 │  ▲
   │ 直传/回放视频    │  │ 账号、视频、任务、修正（PostgreSQL：Neon 或 Supabase）
   ▼                 ▼  │
Cloudflare R2 ◀──── Modal：GPU worker 容器（docker/worker.Dockerfile，按秒计费，空闲自动关机）
（视频 + 分析结果）      ▲
                        └─ API 有新任务时调用 wake 接口启动；每 10 分钟巡检一次兜底
```

| 组件 | 代码 | 本地默认 | 线上 |
| --- | --- | --- | --- |
| 数据库 | `rallyiq/db.py` | SQLite（`data/rallyiq.db`） | PostgreSQL（`RALLYIQ_DATABASE_URL`） |
| 文件存储 | `rallyiq/blobs.py` | 本地 `data/videos/` | S3 兼容对象存储（`RALLYIQ_STORAGE=s3://…`，推荐 R2） |
| 任务队列 | `rallyiq/store.py`、`rallyiq/pipeline/worker.py` | API 进程内一个 worker | PostgreSQL 队列，任意多个 GPU worker |
| 账号 | `rallyiq/api/auth.py` | 关闭（单人本机） | `RALLYIQ_AUTH=on`，邮箱 + 密码，邀请码注册 |
| 网页 | `web/` | `npm run dev` | 打包进 API 镜像，同域名提供（`RALLYIQ_WEB_DIR`） |

本地用法完全不变：不设置这些环境变量时，仍是 SQLite + 本地文件 + 进程内 worker + 无账号。

### 任务队列为什么够生产用

- **多 worker 安全**：PostgreSQL 上用 `SELECT … FOR UPDATE SKIP LOCKED` 领取任务，同一任务不会被两台 GPU 同时处理。
- **租约 + 心跳**：worker 运行任务时每 60 秒续租；GPU 容器崩溃、被回收或超时后，租约过期（默认 5 分钟）的任务会被
  自动放回队列重试，最多 2 次，仍失败则标为失败并写明原因。旧 worker 恢复后写入会被忽略，不会覆盖新结果。
- **按需启动、空闲关机**：`python -m rallyiq.pipeline worker --idle-exit 120` 队列空 2 分钟就退出，GPU 停止计费；
  API 每次排队时 POST `RALLYIQ_WORKER_WAKE_URL` 唤醒（20 秒内最多一次），Modal 上的定时巡检兜底。
- **成本护栏**：每个用户同时排队 + 运行的任务最多 `RALLYIQ_MAX_ACTIVE_JOBS`（默认 3）；单个视频最大
  `RALLYIQ_MAX_UPLOAD_MB`（默认 4096）；Modal 侧最多同时 2 块 GPU（`MAX_GPUS`）。
- 不需要再加 Redis / Celery：单人到小团队规模，数据库队列更简单、少一个要付费和维护的服务。

### 视频存储

- 浏览器直接把视频 PUT 到 R2（预签名 URL，6 小时有效），不经过 API 服务器，大文件上传不受平台请求体限制。
- 回放时 API 返回 307 跳转到 R2 预签名地址，浏览器直接从 R2 按 Range 流式读取；R2 出流量免费。
- 分析结果（JSON）也存在 R2；数据库只记录元数据和"有哪些结果"，列表页不需要逐个查询存储桶。
- 删除视频（网页"删除"按钮或 `DELETE /api/videos/{id}`）会同时删掉 R2 上的原视频、所有结果和修正记录。

### 账号

- 邮箱 + 密码（scrypt 哈希），会话令牌存在 HttpOnly + Secure + SameSite=Lax cookie，`<video>` 和下载链接直接可用。
- 注册方式 `RALLYIQ_SIGNUP`：`invite`（默认，需要邀请码）/ `open` / `closed`；第一个注册的账号是管理员。
- 每个用户只能看到、修改、删除自己的视频；访问别人的视频一律返回 404。
- 登录 / 注册按 IP 限速（10 分钟 10 次）。也可在服务器上建号：`python -m rallyiq.api.auth create-user 邮箱`。

## 2. 推荐的服务和费用估算

| 用途 | 推荐 | 计费方式（概数） | 备选 |
| --- | --- | --- | --- |
| web + API | **Fly.io** `shared-cpu-1x` 512 MB | 常开约 $3/月；配置了无流量自动停机，实际更低 | Render、Railway、任意 VPS |
| 数据库 | **Neon** 免费档 | 0.5 GB 免费，足够存元数据 | Supabase 免费档（一周无访问会暂停） |
| 视频存储 | **Cloudflare R2** | 10 GB 免费，之后约 $0.015/GB·月，下载流量免费 | AWS S3（出流量贵）、Backblaze B2 |
| GPU | **Modal** L4 | 约 $0.80/小时，按秒计费；Starter 档每月送 $30 额度 | Modal T4（约 $0.59/小时）、RunPod Serverless |
| 域名（可选） | 先用免费的 `xxx.fly.dev` | 自有域名约 $10/年 | — |

**单个视频的 GPU 成本（估算）**：在 L4 上预计处理耗时约为视频时长的 0.2–0.5 倍，另有 1–3 分钟冷启动。
一场 1 小时的比赛约 15–35 分钟 GPU，约 **$0.2–0.5**。

**每月总费用（估算，不含域名）**

| 使用量 | GPU（L4） | 存储（R2） | web/API + 数据库 | 合计 |
| --- | --- | --- | --- | --- |
| 轻度：每月 10 小时视频 | $2–5（在 Modal 每月 $30 赠送额度内 ≈ $0） | 约 60 GB 累积，首月 $0.75 | $0–3 | **约 $1–5/月** |
| 中度：每月 50 小时视频 | $10–25（扣除赠送额度后 ≈ $0） | 每月新增约 300 GB，每月多约 $4.5 | $3 | **约 $5–10/月，存储逐月增加** |

长期最大的成本是**原始视频存储**（手机 1080p 约 6 GB/小时）。后续可加：分析完成后转码为低码率回放版、
原片保留 N 天后自动删除（R2 生命周期规则即可，不需要改代码）。

## 3. 上线步骤

需要你本人完成的只有 **注册账号 / 绑卡 / 创建密钥** 这几步（标 ★）；其余命令可以在你的 Mac 上由我代为执行。

1. ★ **Cloudflare R2**：开通 R2（需要绑定付款方式，免费额度内不扣费）→ 创建存储桶 `rallyiq` →
   创建 R2 API 令牌（Object Read & Write，仅限该桶），记下 Account ID、Access Key ID、Secret。
   在桶的 CORS 设置里粘贴 `deploy/r2-cors.json`（把域名改成你的网站地址）。
2. ★ **Neon**：注册 → 新建项目 `rallyiq`（区域选 Asia Pacific / Singapore 或 Tokyo）→ 复制连接串
   （`postgresql://…?sslmode=require`）。表会在 API 首次启动时自动创建。
3. ★ **Modal**：用 GitHub 登录 modal.com。然后在仓库根目录：
   ```bash
   pip install modal && modal setup
   modal secret create rallyiq RALLYIQ_DATABASE_URL='…' RALLYIQ_STORAGE=s3://rallyiq \
       RALLYIQ_S3_ENDPOINT=https://<ACCOUNT_ID>.r2.cloudflarestorage.com RALLYIQ_S3_REGION=auto \
       AWS_ACCESS_KEY_ID='…' AWS_SECRET_ACCESS_KEY='…' RALLYIQ_WORKER_WAKE_TOKEN="$(openssl rand -hex 24)"
   modal deploy deploy/modal_worker.py     # 首次会构建 GPU 镜像并下载模型权重，约 10–20 分钟
   ```
   记下输出里 `wake_app` 的 URL。
4. ★ **Fly.io**：注册并绑卡，安装 `flyctl` 后：
   ```bash
   fly launch --no-deploy --copy-config --config deploy/fly.toml   # 选一个应用名，例如 rallyiq-sam
   fly secrets set --config deploy/fly.toml \
       RALLYIQ_DATABASE_URL='…' RALLYIQ_STORAGE=s3://rallyiq \
       RALLYIQ_S3_ENDPOINT=https://<ACCOUNT_ID>.r2.cloudflarestorage.com \
       AWS_ACCESS_KEY_ID='…' AWS_SECRET_ACCESS_KEY='…' \
       RALLYIQ_INVITE_CODE="$(openssl rand -hex 8)" \
       RALLYIQ_WORKER_WAKE_URL='<第 3 步的 wake_app URL>' RALLYIQ_WORKER_WAKE_TOKEN='<与第 3 步相同>' \
       RALLYIQ_CORS_ORIGINS=https://<应用名>.fly.dev
   fly deploy --config deploy/fly.toml
   ```
5. **验证**：
   ```bash
   python scripts/smoke_online.py --base https://<应用名>.fly.dev --video 一段短视频.mp4 --invite <邀请码>
   ```
   输出 `SMOKE OK` 即全链路（注册 → 直传 R2 → GPU 分析 → 结果 → 回放）正常。然后在网页上用邀请码注册自己的账号。
6. （可选）★ 自有域名：`fly certs add rallyiq.你的域名`，按提示在域名商处加 CNAME；同时更新 CORS 两处配置。

所有变量的说明见 `deploy/env.example`。AI 教练点评默认用 Claude API：在 Fly 上再 `fly secrets set ANTHROPIC_API_KEY=…`
即可（按调用付费，每份报告约几美分；不设置则用离线规则模板）。

## 4. 本地跑一遍线上同款环境

```bash
docker compose up --build            # PostgreSQL + S3（moto，内存存储）+ web/API + 演示 worker（合成数据，无需 GPU）
# 打开 http://localhost:8000 ，用邀请码 rallyiq 注册
docker compose --profile gpu up --build   # 有 NVIDIA GPU 的机器：加上真实 GPU worker
python scripts/smoke_online.py --video clip.mp4 --invite rallyiq
```

CI（`.github/workflows/docker.yml`）在每个相关 PR 上做同样的事：构建镜像、启动整套服务、跑 `smoke_online.py`，
并检查 GPU worker 镜像能构建、依赖能导入；`ci.yml` 的单元测试同时跑 SQLite 和 PostgreSQL 两种后端。

## 5. 运维要点

- **看队列**：`GET /api/health` 返回排队 / 运行中的任务数、数据库和存储类型。
- **卡住的任务**：worker 每次领任务前会把租约过期的任务放回队列；Modal 的 `sweep` 每 10 分钟检查一次，
  有排队任务就启动 GPU。
- **备份**：Neon 自带时间点恢复；R2 可开启对象版本或定期 `rclone sync` 到另一个桶。
- **换 GPU 型号**：`RALLYIQ_MODAL_GPU=T4 modal deploy deploy/modal_worker.py`（更便宜更慢）或 `A10G`（更快）。
- **线上同样可用的功能**：回合片段导出（API 让 ffmpeg 直接按 Range 读取 R2 预签名地址，不下载整段视频）、
  视频质量评估（缓存戳来自对象存储元数据）、难例汇总（普通用户只看自己的，管理员看全部）、赛后报告与 AI 点评；
  个人长期档案的事件日志存进数据库（每个账号一份，只追加），本地旧的 `memory/events.jsonl` 首次使用时自动导入。
- **从本机迁移数据**：本地 `data/videos/` 的目录结构与桶里的键完全一致，可用 `rclone copy data/videos r2:rallyiq/videos`
  上传，数据库行需要另外导入（目前没有自动迁移脚本，有需要时再加）。
