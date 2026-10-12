# 全通道整窗事件实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** 将常规频段扫描、复核和训练统一成唯一的全通道 60 秒窗口及三态多标签。

**Architecture:** 新建 EventWindow、WindowAnalysis、WindowReview 表，保留旧局部事件；一个分析记录也是一个可领取的扫描任务。独立的整窗处理模块复用连续读取和局部筛选，完整证据以有界文件保存；新 API/界面与旧历史入口分离。模型通过统一数组契约和已确认整窗标签训练，旧分类头不冒充新任务。

**Tech Stack:** Python 3.12、NumPy/SciPy、SQLAlchemy/Alembic、FastAPI、React/TypeScript、PyTorch（现有隔离环境）。

**Spec:** `docs/superpowers/specs/2026-09-15-whole-window-events-design.md`

## Global Constraints

- 原工作树 `D:/das-sentinel-runtime/waterfall-range-previews-git-20260901-2/.worktrees/jjk-scan-progress`，分支 `fix/jjk-scan-progress`。
- 保留既有未提交修改；本次不提交、不推送、不合并、不删除旧事件/人工标签/取消任务。
- UTC 整分钟，60 秒，100 Hz，0.5–40 Hz；低频流程不改。
- 数组 `[实际道数,6000]`，全道原始数值及真实距离；站点扫描范围和坏道通过独立掩码表达。
- 标签键固定 `earthquake, blast, traffic`，值 `present, absent, unknown`；仅背景等价于三个 absent，unknown 不作为训练负类。
- 窗口身份仅站点与主窗起止；处理/来源变化只增加分析版本。保存标签必须绑定完整输入签名。
- 旧 ROI 标签只作为线索。没有合格整窗模型时机器类别待定。
- 使用现有运行环境，不全局安装/升级依赖；实际重启和队列迁移前完成备份及映射核验。
- 测试命令：`D:/das-sentinel-runtime/.venv/Scripts/python.exe -m pytest ... -q`；Torch 测试使用 `.venv-first-model` 并设置 PYTHONPATH 为 backend/src。
- 已获设计和实施授权，任务间连续推进。并行任务严格独占文件；root 负责集成和上线。每块独立审查后再关闭。

## 共享契约

`whole_window.py` 导出：

```python
WINDOW_SECONDS = 60
SAMPLE_RATE_HZ = 100
ALGORITHM_VERSION = "whole-window-v1"
def window_start(value: datetime) -> datetime: ...
def window_id(site_id: str, start: datetime) -> str: ...
def inspect_sources(paths: Sequence[Path]) -> list[dict]: ...
def plan_windows(sources: Sequence[dict]) -> list[dict]: ...
def analyze_window(paths: Sequence[Path], start_utc: datetime,
                   profile: ProcessingProfile, output_dir: Path) -> dict: ...
```

Source dict 必含 `path, fingerprint, size_bytes, mtime_ns, start_utc, end_utc, sample_rate_hz, distance_m, unit`；时间为含时区 ISO 字符串。计划项含 `start_utc,end_utc,sources,coverage`，coverage=`complete` 或 `incomplete`；sources 为覆盖主窗及读取上下文的 source dict。主窗之间 `[start,end)` 不重叠。规划器不能跨遥远缺口枚举空白年月；仅枚举与来源相交的分钟。上下文缺口也报告 incomplete。

分析结果 JSON 含 `schema_version:1, algorithm_version, start_utc,end_utc, sample_rate_hz, shape, unit, grid_sha256, input_sha256, recognition_range_m:{start,stop}, sources, features, candidate, score, evidence_count, artifacts`。features 为 `dict[str,float|int|None]`；完整证据为 evidence.json 数组。artifacts 为 `[{kind,path,media_type,sha256}]`，path 是 output_dir 内单一文件名。必须含 waveform.npz、metadata.json、evidence.json、waterfall.png、waveform.png；result.json 最后原子写入，不把自己列入 manifest。NPZ keys=`waveform,distance_m,relative_seconds,valid_channel_mask,recognition_channel_mask,region_id`。不含 ROI 监督。

新 REST 前缀 `/api/v1`：

- `GET /event-windows?site_id=&review_status=unreviewed|reviewed|all&recommended=true&limit=30&offset=0` → `{items,total,limit,offset}`。
- item=`{id,site_id,site_name,start_utc,end_utc,status,score,priority,legacy_count,catalog_matches,analysis_id,input_sha256,latest_review}`；status 为 pending/queued/running/succeeded/failed/cancelled，latest_review 包含 labels、note、created_at、stale。
- `GET /event-windows/{id}` → item + `{features,recognition_range_m,shape,unit,machine_labels,artifacts,legacy_events}`；machine_labels 未验证模型时为空数组。artifact 包含 kind 和受保护 URL。
- `GET /event-windows/{id}/evidence?limit=50&offset=0` → `{items,total,limit,offset}`。
- `POST /event-windows/{id}/reviews` body=`{analysis_id,input_sha256,labels:{earthquake,blast,traffic},note,request_id}` → review。CSRF、幂等、过期输入拒绝 409。
- `GET /window-jobs?site_id=&status=&limit=30&offset=0` → 分页任务；`POST /window-jobs/{id}/cancel`、`POST /window-jobs/{id}/retry`。
- worker: `POST /internal/window-jobs/claim` `{worker_id}` → 204 或 analysis snapshot；heartbeat/fail/result 在 `/internal/window-jobs/{id}/...`。
- claim 必含 `id,window_id,site_id,site_data_root,start_utc,end_utc,profile_payload,sources,attempt_count`。
- heartbeat `{worker_id,attempt_count}`；fail `{worker_id,attempt_count,error_code,error_message}`；result `{worker_id,attempt_count}`（API 从固定 `<app-data>/window-artifacts/<analysis_id>/attempt-<attempt_count>/result.json` 验证并读入，禁止任意路径）。

list/detail 包含 `site_timezone`；图件/证据 URL 绑定 analysis 和 attempt。每次重试单独目录，防迟到结果覆盖。

## Task 1: 整窗读取、检测、测量及图件

**Files:** Create `backend/src/das_sentinel/whole_window.py`, `backend/src/das_sentinel/whole_window_plots.py`; Test `tests/test_whole_window.py`。

**Interfaces:** 输出共享契约的源规划和 analyze_window；不写数据库，不改旧 pipeline。

- [x] 编写失败测试：同站点同分钟不受输入秒数影响；时区归一；跨文件与缺口规划；同窗不相交通道信号只有一个结果且证据全部保留；排除道不影响分类测量但 NPZ/预览原始值仍存在。

```python
def test_identity_ignores_local_peak():
    a = datetime(2023, 9, 27, 3, 2, 1, tzinfo=UTC)
    assert window_id("site", window_start(a)) == window_id(
        "site", window_start(a + timedelta(seconds=1)))
```

- [x] 运行该测试并确认因为缺少新实现失败。
- [x] 使用现有 `read_low_frequency_window(...,sample_rate_hz=100)` 读取全部通道及足够 STA/LTA、0.5 Hz 滤波保护区。主窗精确 6000 点，浮点时钟/文件缺口遵循现读取器验证。检测内部块必须有条件化上下文，证据按稳定 UTC 峰锚点归属唯一主窗。
- [x] 源快照和读取前后变化验证；同窗内网格/单位变化失败。保留原生米坐标，mask 使用 profile 距离范围及真实坏道。JJK 分区边界 299/499 m 仅作物理区域说明；不能用统一道号假定。
- [x] 保留局部融合的相容性与质量门槛；主窗 candidate=至少一个合格局部证据组。计算整窗分频能量分位数/时间变化、有效/活跃比例、空间覆盖、相关/时移与传播、地表/井下能量统计；无效测量 None。
- [x] 图件上下均使用本地绝对参考时间与相对秒。全通道 waterfall、代表波形来自同主窗，不把排除道零值当作原始预览。
- [x] 原子写 NPZ/JSON/PNG 及 sha256；完整 evidence JSON 最大 32 MiB、manifest 最多 8 项；超出明确失败，不截断。
- [x] 运行新测试及 `tests/test_continuous_low_frequency.py tests/test_screening.py tests/test_waveform_samples.py`；记录结果，独立审查。

## Task 2: 窗口持久化、队列和 API

**Files:** Create `backend/src/das_sentinel/persistence/window_models.py`, `backend/src/das_sentinel/persistence/migrations/versions/0003_event_windows.py`, `backend/src/das_sentinel/services/event_windows.py`, `backend/src/das_sentinel/api/routes/event_windows.py`, `backend/src/das_sentinel/api/window_schemas.py`; Modify `backend/src/das_sentinel/api/app.py`（注册路由）； Test `tests/test_event_windows.py`。

**Interfaces:** 消费共享契约；服务导出 `enqueue_windows(session,site_id,profile_id,plans,*,priority=0,catalog_matches=(),legacy_event_ids=()) -> list[WindowAnalysis]`，调用方负责事务；导出 `WindowService(session,artifact_root)` 的 list/detail/review/claim/heartbeat/result/fail/cancel/retry 方法。artifact_root 指向 `<app-data>/window-artifacts`。所有 API 写事务通过现 AsyncWriteCoordinator。

- [x] 先编写隔离 SQLite 测试：迁移保留旧标签；同站点分钟唯一、重复相同计划幂等；分析版本改变不改 window_id；地震+车辆保存；unknown 不派生背景；重复请求返回同记录、重复请求不同内容拒绝。

```python
labels = {"earthquake": "present", "blast": "unknown", "traffic": "present"}
assert saved["labels"] == labels
assert saved["background_only"] is False
```

- [x] 验证测试失败。新增表 EventWindow（唯一 site/start/60s、优先级/目录/旧事件引用）、WindowAnalysis（immutable plan/hash、处理版本、来源、状态/claim/结果）、WindowReview（独立三态、input_sha256、analysis_id、幂等 request_id、管理员、时间）。新增表迁移只做加法，不改旧表数据。窗口列表 SQL 分页，不全量装载波形。
- [x] 完整主窗才能排队；缺口窗口记录为 failed/coverage_incomplete，不能标噪声。retry 只显式作用失败任务；cancel 幂等并阻止迟到结果写入；claim 原子设置 owner/attempt、检查过期心跳，不能恢复 cancelled。
- [x] 实现共享 REST 契约；请求 labels 必须三个键，unknown 可独立保存；只允许成功且与当前输入签名一致的分析被整窗复核。旧 ROI 线索不得写入 WindowReview。
- [x] 结果验证固定目录、常规文件、限制数量/体积、校验 sha256/来源/主窗与计划一致、拒绝 NaN/任意路径；evidence 分页读取有界文件，图件需要登录；不把数组/源绝对路径放入普通列表。
- [x] 至少测试 CSRF、未登录、worker token、跨任务 owner、失效 claim、超限 evidence、结果 hash 不符、过期 review、source/profile 变化后 review stale、分页与优先级。
- [x] 运行 `tests/test_event_windows.py tests/test_review_batches.py tests/test_events_api.py tests/test_persistence.py`（实际测试文件以 rg 确认为准）；生成独立审查记录。

## Task 3: 新扫描与 worker 衔接

**Files:** Create `backend/src/das_sentinel/services/window_planning.py`, `backend/src/das_sentinel/worker/window_main.py`; Modify `backend/src/das_sentinel/services/scans.py`, `backend/src/das_sentinel/worker/main.py`（单项结果 422 错误隔离）； Test `tests/test_window_planning.py`, `tests/test_window_worker.py`, `tests/test_worker.py`。

**Interfaces:** `enqueue_window_files(sessions,site,files,artifact_root)` 验证稳定源后调用 inspect_sources/plan_windows，再调用 Task 2 enqueue_windows。Worker 只消费新 window-jobs 协议；旧 worker 不同时启动。

- [x] 先写测试：同分钟多个文件/重复扫描只创建一个窗口对象；元数据从真实文件读取，跨日上下文可来自已有相邻来源；取消任务不参与旧队列迁移；新扫描不创建局部 Job。
- [x] 默认站点目录扫描在稳定检查后进入窗口规划，保留原 scan 进度接口；源注册沿用 SourceFile fingerprint 约束。NAS 读取不占数据库写锁；最终入库重新核对 profile/source 快照。
- [x] 新 worker 使用 Settings、现有 token header，定时 heartbeat；分析到固定 analysis 目录后只提交 result 引用。可归属单任务的读取/协议大小错误 fail 后继续；401/403、非法 claim 必须停止。

```python
# 第一项格式失败后仍能处理第二项；取消/失去 owner 的结果不提交。
assert outcomes == ["failed", "succeeded"]
assert client.claim_calls == 2
```

- [x] 新旧 worker 均测试 422 result 可隔离；不能吞掉所有 HTTP4xx。默认启动新整窗 worker，日志不含 token。
- [x] 旧 1,151 个 queued v2 文件先形成备份/映射 JSON，完整覆盖映射通过后标明由新分析接管；不删除、恢复或重新处理 cancelled。不自动重跑旧 succeeded 文件，除本次明确选择用于新整窗复核的来源。
- [x] 运行 worker/planning/scans/discovery/jobs 相关测试；独立审查，主代理核实真实映射后执行。

## Task 4: 默认整窗复核与任务界面

**Files:** Create `frontend/src/windowApi.ts`, `frontend/src/pages/WindowEventsPage.tsx`, `frontend/src/pages/WindowEventsPage.test.tsx`, `frontend/src/components/WindowJobs.tsx`, `frontend/src/components/WindowJobs.test.tsx`, `frontend/src/components/window-events.css`; Modify `frontend/src/App.tsx`, `frontend/src/pages/JobsPage.tsx`。

**Interfaces:** 使用共享 REST。`/events` 默认 WindowEventsPage，`/events/legacy` 保留现 EventsPage。JobsPage 顶部整窗任务区，旧任务表明确历史文件任务。保持其他现有功能入口可达。

- [x] 测试先行：相同一分钟只显示服务返回的一个窗口；服务器分页；推荐/全部、站点筛选；打开整窗图件；三个独立三态控件可保存地震+车辆；unknown 不自动变 absent；输入签名变更拒绝保存并重新加载。
- [x] 队列/主图区/特征区域保持现有简洁布局，顶部明确“60 秒全通道窗口”；特征中文；本地时间按现有助手显示，不更改图内相对轴。
- [x] 人工标签每类“未确定/存在/不存在”，提供“仅背景”明确操作一次设三个 absent；显示旧局部标签为历史线索而非已复核。无合格机器模型显示待定。
- [x] 默认推荐按 priority、score 排序，保留 2023/2024 目录批优先；失败/缺数据不能呈现可保存的全窗标签。
- [x] 新任务列表分页、状态筛选、cancel/retry 错误提示；旧文件任务不隐去。生产 API 路由 ready 后运行前端测试、tsc/build，浏览器验证实际数据。

## Task 5: 整窗多标签训练与数据清单

**Files:** Create `backend/src/das_sentinel/classifiers/window_model.py`, `backend/src/das_sentinel/window_training.py`, `backend/src/das_sentinel/services/window_training.py`, `backend/src/das_sentinel/api/routes/window_training.py`, `frontend/src/components/WindowTraining.tsx`, `tests/test_window_model.py`, `tests/test_window_training.py`; Modify `frontend/src/pages/ModelsPage.tsx` 与对应测试。新增路由注册由 root 修改 api/app.py，避免 Task 2 文件冲突。

**Interfaces:** 消费 Task 1 NPZ + metadata，Task 2 WindowReview；CLI `python -m das_sentinel.window_training export --database-url ... --artifact-root ... --output ...`；`train --manifest ... --output ...`。导出只包括当前匹配签名且三态至少一类已知的完整窗口，保留日期/源上下文/目录震源/grid。

- [x] 先写测试：全窗特征受 ROI 以外有效信号影响；排除道变化不影响；可变 H 和物理分区；同源/日期/目录震源不跨划分；未知类别不参与 loss/metrics；旧 ROI 不能导成已标整窗。
- [x] PyTorch 轻量卷积，信号及 mask 输入平面，真实分区平均/最大池化，输出3个独立 logits；逐窗或同网格 batch 避免 padding 污染 GroupNorm。预处理与推理共享，训练只用一个经验证网格，产物绑定 grid/preprocess/label schema。

```python
known = targets != -1
loss = (binary_cross_entropy_with_logits(logits, targets.clamp_min(0),
        reduction="none") * known).sum() / known.sum().clamp_min(1)
```

- [x] 同清单同时训练整窗统计特征基线；固定分组 train/dev/test，训练归一化、类别权重/阈值不读取 test；缺少独立组/每类已知正负时明确退出并生成就绪度报告，不能用旧标签或自造标签凑训练。
- [x] 模型保存安全权重+JSON元数据，加载校验哈希/schema/grid；旧 ROI 头不复用。CLI train 生成真实多标签指标，候选未达标不发布。SSL 初始化对照若使用，审计预训练来源；不必为本次额外发起自监督训练。
- [x] 实际导出当前可用新整窗标签；若为零，记录真实欠缺并在模型页展示需要整窗复核的计数；运行合成多标签数据训练只作代码验证，不称站点训练模型。
- [x] 模型页默认展示整窗训练就绪度和整窗训练入口，旧局部模型保留在明确的历史区域，不能让默认训练按钮继续训练旧RF却表示整窗。`GET /window-training/readiness?site_id=` 返回 `site_id,window_count,reviewed_windows,eligible_windows,classes,ready,reasons`；每类计数 present/absent/unknown。`POST /window-training/jobs` body `{site_id}` 在未就绪时409并给原因，否则202后台训练；`GET /window-training/jobs?site_id=` 返回最近20项状态/指标。导出/训练产物保存在 app-data/window-training，不覆盖旧模型。
- [x] API训练服务沿用项目有界后台任务模式，一个训练任务运行；子进程使用环境 `DAS_TRAINING_PYTHON` 指定已有隔离解释器（默认sys.executable），无shell参数拼接。创建前冻结清单、标签及分组；身份参数校验、CSRF、失败恢复与重复点击有可验证行为。缺少Torch或样本不足都明确显示未就绪/失败，不静默回退为ROI训练。
- [x] Torch 环境执行相关测试，普通环境使用 importorskip 保持服务无重依赖；独立审查。

## Task 6: 真实迁移、上线与整体审查

**Files:** Create `docs/whole-window-events.md`; Update 当前计划及 `.superpowers/sdd/2026-09-15-whole-window-events/progress.md`；操作脚本/备份在 `D:/das-sentinel-runtime/manual-work/whole-window-rollout-20260915`。

- [x] 改代码前记录 dirty 文件 sha256/拷贝、HEAD、实际进程、只读 DB 备份；任何部署前再次备份并保留旧启动脚本。
- [x] 将旧 ROI 按唯一 UTC 主窗映射，关联所有相交的旧标签（只作线索）、目录匹配和优先级。先真实读取 2023-09-27 六个候选分钟及实际所需上下文，逐个给出成功/缺口/网格异常，不伪称全部可用。
- [x] 冻结旧队列映射：有效 source/profile/cancelled 检查、窗口计数、完整覆盖、去重、回退信息。只有映射成功且存在新任务的旧 queued 才标记接管，失败者保留可见错误；旧取消/人工标签逐行核对未变。
- [x] 数据库副本先升级和验收，通过再在线备份、短暂切换 API/前端到新构建；先验证 health/auth/read-only API，再启动一个新 window worker 处理已映射队列。
- [x] 浏览器验证 /events 整窗队列、60s全距图、排除区可见、三态多选、历史入口；不伪造生产人工标签。API 测试环境验证保存/幂等/过期错误。
- [x] 运行所有本次新增及受影响测试、前端完整测试/build、Ruff、git diff --check；整体独立审查修正阻塞问题。
- [x] 验收模型就绪度和导出：明确已有整窗标签数量，数据不足则完成训练能力交付但不宣称训练成功；保留独立模型候选要求。
- [x] 写最终报告：新增窗口/旧映射/失败缺口/排队状态、验证证据、现有数据保留核验、模型限制、未推送 GitHub。

## 完成记录

2026-09-15 已完成Task1–6及本地上线。最终审查和测试证据见 .superpowers/sdd/2026-09-15-whole-window-events/progress.md 与 docs/whole-window-events.md。后台窗口队列继续运行；当前整窗人工标签为0，因此交付训练能力和就绪度，不声称训练出新站点模型。未提交/推送/合并GitHub。
