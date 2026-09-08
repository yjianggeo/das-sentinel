# DAS Sentinel 运维手册

## First boot and NAS security

1. Copy .env.example to .env, generate both secrets, and set DAS_HOST_DATA_ROOT.
2. Give the container user read-only access to raw DAS data and read/write access only to the app-data volume.
3. Start API: docker compose -f compose.yaml -f compose.cpu.yaml up -d api.
4. Verify /healthz, then initialize admin with docker compose ... exec api das-sentinel admin init.
5. Start worker and configure the NAS HTTPS reverse proxy to 127.0.0.1:DAS_HTTP_PORT.
6. Rotate secrets by stopping the stack, editing .env, and restarting; existing sessions are invalidated.

`/data` is excluded from writes and database placement. Place SQLite databases and all application artifacts only in the `/app-data` volume.

## 数据布局

| 路径 | 内容 | 读写性 |
| --- | --- | --- |
| `/data` | 原始 DAS 采集文件（ZD/PRODML HDF5） | **只读**，任何组件不得写入或删除 |
| `/app-data` | SQLite 数据库、Artifact Store、模型 bundle | 读写，需纳入备份 |

Compose 部署中 `/data` 以 `read_only: true` 绑定挂载；SQLite 只落在 `/app-data`。
原始数据的保护由两层保证：挂载只读 + worker 只以 `rb` 模式打开源文件并校验指纹。

## 备份

备份只需覆盖 `/app-data` 卷：

```bash
docker run --rm -v das-sentinel_app-data:/data -v "$PWD:/backup" alpine \
  tar czf /backup/app-data-$(date +%Y%m%d).tgz -C /data .
```

SQLite 在备份前建议执行 `PRAGMA wal_checkpoint(TRUNCATE);` 或在低峰期停止 api 与
worker 后冷备份。恢复时将归档解回同名卷再启动即可。

## 重启恢复

- API 启动时先执行 Alembic 迁移，再扫描 `running` 状态但心跳超时的任务并回收为
  `queued`（僵尸任务回收），无需人工干预。
- worker 崩溃只影响当前认领的文件：该任务心跳超时后由 API 回收重排。
- 容器编排使用 `depends_on: service_healthy`，worker 在 API 健康前不会启动。

## JJK 目录扫描

1. 打开 **任务 → 选择目录并扫描站点**，选择目标站点。
2. 进入具体的 JJK / 日期目录（例如 `jjk20240608/20240610`），选择当前目录，
   确认页面显示的站点根目录相对路径。必须选择非根子目录，不可省略扫描范围。
3. 只点击一次 **开始扫描**。
4. 等待状态依次经过 **正在扫描 → 请等待稳定性复检 → 正在复检 → 创建任务 → 完成**。
   页面显示文件计数与进度；稳定性等待和第二次检查均由服务端自动完成。
5. 扫描期间按钮禁用，不要重复点击或另行发起第二次扫描。刷新页面后可恢复查看
   仍在运行的扫描。
6. 扫描状态仅保存在 API 进程内存中，API 重启后不能续接原扫描；若因重启或 NAS
   断连中断，请在 API / NAS 恢复连接后重新选择目录并启动新扫描。

原始数据全程只读；扫描仅递归处理选定目录内支持的稳定文件。一个目录含有许多
分钟文件时，每个稳定且受支持的文件会创建或复用一个分析任务，不会把整个目录
合成一个任务。生产目录的验收应只浏览和选择范围，未经确认不要点击开始扫描，
以免批量入队。

CLI 同样必须提供站点根目录相对的 `PATH`，一次命令自动等待两次检查结束：

```bash
das-sentinel scan <site-id> <relative-path>
das-sentinel scan <site-id> jjk20240608/20240610
```

## 事件图件临时预览

打开一个已有事件后，页面会保留该事件分析时生成的原始 PNG，不会覆盖或改写历史
图件。距离输入框默认显示事件通道对应的自动距离范围；不修改输入框时，点击一次
**生成临时预览**即按事件通道自动生成。只要改过任一输入框，页面就会按手动范围
处理；此时起始、结束距离必须同时保留或填写为可用范围内的有效数值，且起始距离
严格小于结束距离，再点击一次生成。

生成完成后，可在**瀑布图**、**滤波瀑布图**和**波形**三个标签之间切换；一次生成
会得到共用同一个真实数据时间窗口的三张临时图，页面会提示“当前显示临时预览”。
**下载**始终下载当前标签正在显示的图；在上述三个标签中显示临时预览时，下载的
就是对应临时图。PSD、STFT 和事件特征仍使用原分析结果，不随临时预览改变。点击
**重置临时预览**会恢复自动距离输入和原始历史图件；此后不再编辑直接生成时，仍
采用自动距离范围。

源文件长度允许时，三张临时图至少显示 3 秒真实数据；若事件及原有上下文更长，则
保留更长窗口。后续新分析正式生成的瀑布图、滤波瀑布图和波形也沿用这一同窗规则；
已有事件的历史原图保持原样。
窗口碰到当前源文件的开头或结尾时，会尽量把缺少的时长移到另一侧。若整个源文件
不足 3 秒，图中会标注 `Source shorter than 3 s; showing actual duration`，并只显示
实际可用时长，不补造样本，也不跨文件拼接。图题中的 `Display samples` 和
`duration` 描述本次显示窗口及其真实时长，不是事件持续时间；滤波图中的
`Band-pass` 以及页面显示的滤波频带表示本次滤波瀑布图实际采用的频率范围。

时间轴上方的 `Local reference` 是站点时区（例如 `Asia/Shanghai`）的整秒参考时刻，
下方 `Time relative to reference (s)` 是相对该时刻的精确秒数。例如第一条显示样本
发生在整秒参考时刻之后 0.234 秒，该样本的横坐标就是 `0.234` s；自动刻度会覆盖
这个时间范围，但不保证恰好把 `0.234` 标成一个刻度。相对时间的零点只是上方参考
时刻，不一定是事件开始时刻；事件位置以图中的事件标记为准。

## 失败文件重试

- 单文件失败被隔离：`error_code` / `error_message` / `retry_eligible` 记录在任务上，
  队列中其余文件继续处理（见 `tests/test_fault_injection.py` 的验收用例）。
- 源文件在入队后被改动（指纹不一致）会以 `invalid_claim` 失败且**不可重试**——
  这是数据完整性保护，需重新扫描让新指纹入队。
- 可重试失败（如瞬时 I/O）通过 CLI 重排：

```bash
das-sentinel jobs list --status failed
das-sentinel jobs retry <job-id>
```

## 健康检查

- API：`GET /healthz`（Compose 已配置 15 s 间隔健康检查）。
- 本地诊断：`das-sentinel doctor` 检查 CPU/CUDA 后端、存储可写性、数据库启动门禁
  与配置完整性；缺失密钥会警告但不中断诊断本身。

## 端到端验收

`e2e/` 下的 Playwright 用例需要一个播种好的本地栈。`e2e/seed_stack.py` 会创建
隔离数据目录（默认 `%TEMP%\das-sentinel-e2e`，可用 `E2E_STACK_DIR` 覆盖——SQLite
不允许网络路径），写入两个发生时间相差一小时的合成 STA/LTA HDF5，播种管理员
（密码 `test-admin-password`）、站点与处理档案，启动 API（127.0.0.1:8000）并用
内嵌 worker 处理队列：

```bash
python e2e/seed_stack.py        # 保持运行
cd e2e && npm install && npx playwright install chromium
npx playwright test             # E2E_BASE_URL 默认为 http://127.0.0.1:8000
```

## 性能基准

```bash
python scripts/benchmark_baseline.py --device cpu --out docs/acceptance-report.md
```

脚本生成（或复用）`1000 通道 × 1000 Hz × 60 s` 的合成文件，记录墙钟时间、峰值
RSS、后端选择与重复运行一致性。验收标准：指定 4-CPU 验收机上峰值 RSS < 4 GB、
基线平均处理 < 60 s。

## Phase 1 deployment and recovery boundaries

For a CPU/NAS first boot, create a non-example `.env`, start the API, and
verify the JSON health response before creating the administrator and starting
the worker:

```bash
cp .env.example .env
# Set DAS_HOST_DATA_ROOT plus freshly generated DAS_SESSION_SECRET and DAS_WORKER_TOKEN.
docker compose -f compose.yaml -f compose.cpu.yaml up -d --build api
curl -fsS http://127.0.0.1:8000/healthz    # {"status":"ok"}
docker compose -f compose.yaml -f compose.cpu.yaml exec api \
  das-sentinel admin init --password '<admin-password>'
docker compose -f compose.yaml -f compose.cpu.yaml up -d worker
```

The GPU image can be built independently of a GPU runtime:

```bash
docker build -f docker/Dockerfile.gpu -t das-sentinel-gpu:phase1 .
```

That build is not CUDA evidence. Only a target NVIDIA host with a compatible
driver, NVIDIA Container Toolkit, and CuPy can run the runtime smoke check:

```bash
docker run --rm --gpus all --entrypoint python das-sentinel-gpu:phase1 \
  -c "from das_sentinel.signal.backend import smoke_test_cuda; smoke_test_cuda()"
```

Published models are trained from reviewed events, validated, then explicitly
published. A newly claimed job binds the current published model identity; its
event exposes the resulting `fusion` classification method alongside the other
machine decisions. If a job fails with `invalid_model`, repair and publish a
valid model, then rescan or re-enqueue the source file. The failed claim is not
automatically retried because its bound model identity is immutable.

A review `request_id` is idempotent per event: the initial request creates one
audit row (`201`), while a concurrent or later replay returns that same row
(`200`) without appending a second row. A new request ID intentionally appends
another review record.

The React Router advisory concerning unstable React Server Components APIs does
not apply to the current BrowserRouter plus REST client architecture. Do not
introduce RSC APIs before upgrading the affected React Router dependency.
