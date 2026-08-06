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
