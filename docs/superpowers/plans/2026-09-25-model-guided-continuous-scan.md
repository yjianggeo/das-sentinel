# JJK 模型直接扫描与提速 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 使用现有300 m模型扫描Z盘剩余JJK数据及传统规则跳过窗口，只将模型入选窗口加工成可人工复核的30秒样本，并实测提高处理速度。

**Architecture:** API负责持久任务、所有权和结果事务；独立扫描worker负责目录发现、连续读取和阳性加工，常驻Torch子进程负责冻结模型推理。阴性仅保留扫描回执；阳性通过现有完整分析验证后进入独立“模型推荐”范围。先验证逐窗读取的正确流程，再开启有界分块读取。

**Tech Stack:** 现有Python、FastAPI、SQLAlchemy/Alembic、SQLite、NumPy/SciPy/h5py、CPU PyTorch、React/TypeScript/Vitest；不新增第三方依赖。

**Spec:** [已批准设计](../specs/2026-09-25-model-guided-continuous-scan-design.md)。执行者必须先读设计及本计划。

状态：用户已批准当前任务内实施＋整体独立审查。任务 1–7 已实现并验证；任务 8 已完成备份、迁移、本地发布、四批限定试扫验收和 12 个正式目录批次提交，扫描在后台继续，未宣称全盘扫描完成。具体证据与运行状态见 `.superpowers/sdd/2026-09-25-model-guided-continuous-scan/progress.md`。

## Global Constraints

- 沿用 `D:/das-sentinel-runtime/waterfall-range-previews-git-20260901-2/.worktrees/jjk-scan-progress`、`fix/jjk-scan-progress`；保留既有修改，不切主分支、不reset、不自动提交、推送或合并。按任务保存差异检查与验证记录，代替技能默认的逐任务提交。
- 固定模型 `80681390-1d01-492a-8be5-306c6636220d`；阈值 `0.03820184990763664`；不训练、不切换在线模型、不根据试扫表现改变阈值。
- 30秒UTC半分钟对齐，100 Hz、每道3000点；抗混叠降采样原始波形，834道统一网格，≥300 m裁剪后760道，之后才归一化；图件仍全通道。
- 保留单位、真实上下文、时钟单调性、接缝、缺失通道掩码、原始文件稳定性检查；无法判断不等于模型阴性。
- 用户范围为未扫描数据＋传统规则跳过窗口。12个目录白名单照设计原文；已有人工复核不改写，431个明确删除窗口排除，明确取消的旧任务不恢复。
- API/reader Python：`D:/das-sentinel-runtime/.venv/Scripts/python.exe`；Torch Python：`D:/das-sentinel-runtime/.venv-first-model/Scripts/python.exe`。当前16 GB内存、CPU推理，模型2线程起步。
- 连续核心块≤5分钟、reader输出<256 MiB；预取最多1块；不全盘复制NAS、不并行启动多个NAS读取worker。
- 新后台进程隐藏启动；原始数据只读，阴性临时数组及时回收，绝不删除原始文件或旧图件。数据库写入经API单写协调器，worker不直接写数据库。

## Review Focus

1. 目录枚举中途崩溃、结果已提交但响应丢失：重启不得漏源/漏窗/重复入选；由任务1、2、4测试。
2. 两份重叠来源分别相同或内容冲突，或NAS读取中源发生变化：只有确认副本才能合并，冲突与变化不能当阴性；由任务2、3测试。
3. 旧规则跳过窗被模型选中，或推理期间用户新增人工标签/取消任务：允许前者进入复核，保护后者，最终事务再次检查；由任务4测试。
4. 分块边界、缺失尾道及接近阈值的得分：网格/掩码不能漂移，模型分类不能因优化产生未解释变化；由任务3、7测试。
5. 批次期间切换站点在线模型、API/子进程重启、图件失败：显示冻结模型，保留入选回执，不能假报完成；由任务1、3、4、6测试。

## 文件与接口边界

本计划中相对路径均以现有工作树为根。新文件只负责新模式，避免重构旧扫描/训练流程。

|文件|职责|
|---|---|
|`backend/src/das_sentinel/persistence/model_scan_models.py`|批次、工作单元、来源、回执四张表|
|`backend/src/das_sentinel/persistence/migrations/versions/0007_model_scans.py`|新增表及索引；不重写旧记录|
|`backend/src/das_sentinel/model_scan_contract.py`|新模式的JSON契约、指纹和状态常量，禁止导入Torch|
|`backend/src/das_sentinel/services/model_scans.py`|短事务、领取/心跳、去重、统计及结果阶段转换|
|`backend/src/das_sentinel/services/model_scan_discovery.py`|白名单发现、来源索引及有界计划；NAS工作在事务外|
|`backend/src/das_sentinel/model_scan_input.py`|原始整窗输入、统一网格与临时数组，不筛选/绘图|
|`backend/src/das_sentinel/model_scan_inference.py`|常驻Torch进程及逐行协议|
|`backend/src/das_sentinel/worker/model_scan_main.py`|HTTP客户端、独立扫描worker和子进程生命周期|
|`backend/src/das_sentinel/api/model_scan_schemas.py`、`api/routes/model_scans.py`|管理员/worker边界校验、路由|
|`frontend/src/modelScanApi.ts`、`components/ModelScans.tsx`|批次列表、进度、操作和预计时间|
|`tools/benchmark_model_scan.py`|只读输入、写独立输出目录的对照基准|
|`docs/model-guided-scanning.md`|操作、限制、实测结果与续扫说明|

需小范围修改：`api/app.py`注册路由；`services/event_windows.py`和`api/routes/event_windows.py`接入模型推荐及阳性加工保护；`window_grid.py`抽出等价内存映射；`readers/continuous_low_frequency.py`可选有界复用；`frontend/src/windowApi.ts`、`pages/WindowEventsPage.tsx`、`pages/JobsPage.tsx`。旧单窗推理API不替换。

公共数据结构以JSON兼容字典表示，由Pydantic在HTTP边界验证。`ModelSnapshot`完整保存现有deployment字段；新增`protocol_version="model-scan-v1"`。`WindowPlan`使用现有 `plan_windows()` 的起止、sources、coverage，另保存模型输入协议摘要；规则处理profile也冻结。

```python
def inference_key(site_id, start_utc, sources, model, input_protocol):
    from das_sentinel.services.event_windows import digest
    return digest({
        "site_id": site_id, "start_utc": start_utc, "window_seconds": 30,
        "sources": sources, "input_protocol": input_protocol,
        "model_id": model["id"], "weights_sha256": model["weights_sha256"],
        "metadata_sha256": model["metadata_sha256"], "threshold": model["threshold"],
    })
```

来源按规范化绝对UNC路径排序，包含稳定fingerprint/size/mtime、覆盖和网格身份；重复副本另经内容校验。`inference_key`不是“内容一样”的证明，也不替代实际输入数组校验。

## Task 1：持久批次、工作单元及幂等回执

**Files:** 新增上述persistence两文件、`model_scan_contract.py`、`services/model_scans.py`；新增 `tests/test_model_scans.py`。迁移接在`0006_window_skips`之后，在新迁移中直接导入新表，沿用0003的`table.create(checkfirst=True)`模式注册到Base.metadata。

**Interfaces:** `ModelScanService(session, app_data_dir)`；`create(site_id, profile_id, request_id, directories, model, exclusions, *, time_ranges=None) -> dict`；`claim(worker_id, kind=None) -> dict | None`；`heartbeat(unit_id, worker_id, attempt_count) -> dict`；`control(scan_id, action) -> dict`；`get(scan_id) -> dict`；`record_prediction(unit_id, worker_id, attempt_count, result) -> dict`。service调用者拥有事务，service不在构造器恢复/改写状态。time_ranges用于限定试扫的UTC范围，缺省覆盖所选目录全部有效时间；只能缩小范围，不能绕过目录白名单。

- [ ] **1. 写失败测试。** 复用`tests/test_event_windows.py`的`context` fixture及新服务，测试重复创建、错误所有者、过期attempt及暂停/取消。测试模型快照使用固定伪哈希，不读取生产模型；真实deployment校验归任务5。

```python
from tests.test_event_windows import context
from das_sentinel.services.model_scans import ModelScanService

def test_create_replay_keeps_frozen_model(context, tmp_path):
    session, site, profile, _ = context
    service = ModelScanService(session, tmp_path)
    model = {"id": "fixed-model", "weights_sha256": "a" * 64,
             "metadata_sha256": "b" * 64, "threshold": 0.04}
    args = (site.id, profile.id, "request-1", [str(tmp_path)], model, [])
    first = service.create(*args)
    second = service.create(*args)
    assert first["id"] == second["id"]
    assert service.get(first["id"])["model"] == model
    assert service.get(first["id"])["counts"]["judged"] == 0
```

- [ ] **2. 运行红灯。** `& D:/das-sentinel-runtime/.venv/Scripts/python.exe -m pytest tests/test_model_scans.py -q`；新模块不存在应失败，不能把环境缺包当功能失败。
- [ ] **3. 实现最小表与事务。** `ModelScan`保存site/profile、request_id、规范范围、冻结配置、排除清单、状态/时间；唯一键`(site_id,request_id)`，相同request不同payload返回409。`ModelScanUnit`保存scan_id、kind、稳定unit_key、payload_json、status、worker_id、attempt_count、heartbeat和error；唯一键`(scan_id,kind,unit_key)`。kind为`discover/infer/materialize`，不将每个阴性窗建成旧式analysis。`ModelScanSource`保存scan_id、规范path、fingerprint、metadata_json、start/end/发现单元，时间范围建索引。`ModelScanReceipt`保存scan_id、UTC窗起点、inference_key、status、score、input摘要、window/analysis绑定、错误、计时和完成时间；批内唯一`(scan_id,start_utc)`，inference_key另建索引用于跨批复用。
- [ ] **4. 实现状态机。** 批次`queued/running/paused/cancelled/succeeded/completed_with_errors/failed`；单元`queued/running/succeeded/failed/cancelled`；回执`negative/selected/pending_artifacts/ready/unjudgeable/excluded/protected/cancelled`。推理错误与模型阴性分开。暂停拒绝新领取，允许已领取单元提交；取消使未完成领取失效，拒绝迟到结果，不删除已经完成的人工结果。默认180秒租约、15秒心跳；过期attempt最多自动重领3次，之后失败；显式retry只重置失败阶段，不恢复cancelled。

```python
from sqlalchemy import update

changed = session.execute(update(ModelScanUnit).where(
    ModelScanUnit.id == unit_id,
    ModelScanUnit.status == "queued",
).values(status="running", worker_id=worker_id,
         attempt_count=ModelScanUnit.attempt_count + 1, heartbeat_at=now))
if changed.rowcount != 1:
    raise WindowError("claim_lost")
```

- [ ] **5. 加入可执行断言并绿灯。** 同一result重放返回原回执；修改score/输入后重放拒绝；NaN/Inf/范围外score拒绝；threshold由冻结模型取值；阳性定义`score >= threshold`。阴性插入不能增加EventWindow、WindowAnalysis或WindowReview行。事务回滚后重试可成功。迁移在临时数据库从0006升级、二次升级、foreign_key_check验证；没有drop旧表的downgrade。
- [ ] **6. 保存验证。** 运行`tests/test_model_scans.py tests/test_persistence.py`；只记录本任务diff，不提交已有脏文件。

## Task 2：可恢复发现、来源去重及窗口计划

**Files:** 新增`services/model_scan_discovery.py`、`tests/test_model_scan_discovery.py`；扩展`services/model_scans.py`来源页/完成事务。复用`services/discovery.py`、`services/window_planning.py`，不重写旧scan service。

**Interfaces:** `discover_directory(root: Path, directory: Path, *, stable_seconds=10.0) -> Iterator[dict]`输出分批来源、子目录及明确错误；`plan_source_windows(sources: list[dict], *, block_seconds=300) -> list[dict]`输出有界连续块，每块含原始WindowPlan；`commit_discovery_page(unit_id, worker_id, attempt_count, page_index, payload) -> dict`、`finish_discovery(unit_id, worker_id, attempt_count, page_count, manifest_sha256) -> dict`；所有页先落盘，目录完成必须校验连续页及摘要。

- [ ] **1. 写并运行失败测试。** 使用临时HDF5而非Z盘；复用`tests/test_whole_window.py`的`write`和`BASE`生成连续分钟文件、重复副本、边界缺失文件。

```python
from tests.test_whole_window import write, BASE
from das_sentinel.whole_window import inspect_sources
from das_sentinel.services.model_scan_discovery import plan_source_windows

def test_blocks_plan_each_half_minute_once(tmp_path):
    paths = [write(tmp_path / "a.h5", -60, 120),
             write(tmp_path / "b.h5", 60, 120)]
    blocks = plan_source_windows(inspect_sources(paths), block_seconds=60)
    plans = [plan for block in blocks for plan in block["windows"]]
    starts = [plan["start_utc"] for plan in plans]
    assert len(starts) == len(set(starts))
    assert BASE.isoformat() in starts
    assert all(len(block["windows"]) <= 2 for block in blocks)
```

- [ ] **2. 实现目录发现。** 规范Z盘为站点UNC根；路径必须位于批准目录内，沿用anchored source/reparse防护。目录名仅作定位提示，时间/采样率/距离从HDF5读取；页上限256条，文件大小与mtime至少间隔10秒两次观察。缺失、变化、格式不支持各自记录，不把一直变化的文件算完成。快照/回收站/工程目录不进入递归；`eq`不新发现，已有登记来源参与相邻上下文与重复检查。
- [ ] **3. 实现持久发现闭环。** 来源页及子目录单元同事务登记，重复页内容相同可重放、不同拒绝。目录完成只在所有页摘要验证后提交；崩溃后未完成目录重新核对，可增补遗漏页，不能用内存文件索引宣布完成。本批次发现结束后新增文件留待下一次批次；重新批次仍依源身份去重。
- [ ] **4. 实现来源索引与计划。** 先为一个顶层目录完成索引，再按UTC覆盖查询构建≤5分钟块；若配置time_ranges，只计划完整位于相应区间的30秒核心窗，上下文可读取区间外的批准来源。跨目录上下文使用已有来源索引和现有`_neighbor_snapshots`。全盘批次在所有目录发现结束时，再补查早期因尚未发现邻接来源而不完整的边界窗。未完整的窗记`unjudgeable`，不创建普通复核事件；此前431排除身份始终优先。按时间重叠筛来源，禁止每窗遍历全部历史SourceFile。

```python
margin = timedelta(seconds=READ_CONTEXT_SECONDS)
overlap = select(ModelScanSource).where(
    ModelScanSource.scan_id == scan_id,
    ModelScanSource.start_utc < block_end + margin,
    ModelScanSource.end_utc > block_start - margin,
)
```

- [ ] **5. 加入故障与重复测试。** 枚举第2页前崩溃后所有源仍可找回；同一页响应丢失重放不重复；目录时间名错误仍按HDF5时间计划；相同字节副本合并，不同内容但相同时间标冲突；源验证后变更拒绝复用；610条规则skip不当作模型完成，排除清单身份不生成infer单元。重复内容校验仅用于发生重叠的来源，不对全部文件预先全文件哈希。
- [ ] **6. 运行绿灯。** `tests/test_model_scan_discovery.py tests/test_window_planning.py tests/test_scan_service.py`；在临时目录验证，不全盘跑发现。

## Task 3：无图件输入准备与常驻推理

**Files:** 新增`model_scan_input.py`、`model_scan_inference.py`、`tests/test_model_scan_input.py`、`tests/test_model_scan_inference.py`；修改`window_grid.py`抽出内存映射供旧normalize_rows与新路径共用；不改模型架构或权重。

**Interfaces:** `read_scan_input(plan: dict, model_metadata: dict, cache_root: Path) -> tuple[dict, dict]`返回数组字典及来源/数组摘要；`normalize_arrays(arrays: dict, geometry: dict, fixed_grid: dict, recognition_range: dict) -> dict`返回6个既有数组；`ResidentPredictor(folder: Path, model: dict).predict(arrays: dict) -> dict`输出score/threshold/label/model_id；CLI `python -m das_sentinel.model_scan_inference serve --model-folder PATH --snapshot PATH --spool-root PATH`。

- [ ] **1. 写输入等价失败测试。** 以现有`analyze_window(skip_empty=False)`产生的waveform为参照，分别用新reader和`normalize_prediction_row`比较；覆盖三种原生距离网格、缺失尾道、常量通道、不同单位与跨文件接缝。数组关键比较：

```python
for key in ("distance_m", "relative_seconds", "valid_channel_mask",
            "recognition_channel_mask", "region_id"):
    np.testing.assert_array_equal(actual[key], expected[key])
np.testing.assert_allclose(actual["waveform"], expected["waveform"], rtol=0, atol=0)
assert actual["waveform"].shape == (834, 3000)
```

这里逐窗基线要求精确复现；任务7才评估分块的浮点容差。为测试构造训练grid为834道的合成资料，不能对任意小fixture强断言834。
- [ ] **2. 实现输入准备。** 用现有`read_low_frequency_window(paths, start_utc, end_utc, None, None, margin_seconds=30, sample_rate_hz=100)`取得原始30秒主窗与真实上下文，做与`whole_window.py`同样的稳定性/几何/finite/valid检查。将`normalize_rows`中的映射提取为纯内存函数，并让旧函数调用它；不插值距离、不把缺失道标valid。新路径不调用`screen_block`、`_measure`或`create_visualization`。对6个标准数组按名称、dtype、shape及C连续字节计算`model_input_sha256`，连同单位、UTC窗起点与协议摘要保存；此字段不能冒充现有analysis的`input_sha256`。

```python
raw = read.block.data[read.analysis_start:read.analysis_stop].copy()
valid = np.ptp(raw, axis=0) > 0
arrays = {"waveform": raw.T,
          "distance_m": read.block.metadata.channel_distance_m.copy(),
          "relative_seconds": np.arange(3000) / 100,
          "valid_channel_mask": valid}
arrays = normalize_arrays(arrays, geometry, metadata["training_grid"],
                          {"start": 300.0, "stop": None})
```

- [ ] **3. 写常驻推理失败测试。** 重用`tests/test_window_inference.py`的`candidate` fixture；同一输入2次仅加载一次权重，和现有`predict_request`得分相等；模拟修改active指针不改变resident模型。

```python
def test_resident_loads_once(candidate, monkeypatch):
    from unittest.mock import Mock
    from das_sentinel import window_benchmark
    from das_sentinel.model_scan_inference import ResidentPredictor
    folder, _, _, row, _, meta = candidate
    loader = Mock(wraps=window_benchmark.load_candidate)
    monkeypatch.setattr(window_benchmark, "load_candidate", loader)
    snapshot = {"id": "model", "metadata_sha256": file_sha(folder / "model.json"),
                "weights_sha256": meta["weights_sha256"], "threshold": 0.2}
    predictor = ResidentPredictor(folder, snapshot)
    with np.load(row["waveform"], allow_pickle=False) as arrays:
        first, second = predictor.predict(arrays), predictor.predict(arrays)
    assert first["score"] == second["score"]
    assert loader.call_count == 1
```

测试导入`numpy as np`、`file_sha`及fixture；真实文件哈希检查沿用`checked_metadata`。
- [ ] **4. 实现常驻协议。** 启动时checked_metadata+load_candidate一次，torch2线程+inference_mode，调用原`candidate_inputs`。stdin每行JSON含request_id、spool中数组路径及SHA，stdout只输出一行同ID结果；日志走stderr。`np.savez`临时文件不压缩、原子替换且最多2个待消费输入；只接受spool下正规文件、allow_pickle=False并检查SHA。父进程同时排空stderr，单请求120秒超时；进程退出/超时是失败而非阴性，重启进程后仅重试未完成回执。已收到结果后即可清除阴性临时文件，API提交只需结果摘要；响应丢失可幂等重发。
- [ ] **5. 验证。** 主环境运行`tests/test_model_scan_input.py tests/test_window_grid.py tests/test_whole_window.py`；Torch环境运行`tests/test_model_scan_inference.py tests/test_window_inference.py tests/test_window_distance.py tests/test_window_resnet.py`。测试截断协议、请求ID不匹配、NaN、坏数组SHA、单位不符和spool路径越界，不允许Torch测试因缺包全部skip却报告完成。

## Task 4：worker闭环、阳性加工与人工结果保护

**Files:** 新增`worker/model_scan_main.py`、`tests/test_model_scan_worker.py`；扩展`services/model_scans.py`；修改`services/event_windows.py`的入队/领取/结果小范围接入；新增`tests/test_model_scan_materialization.py`。

**Interfaces:** `HttpxModelScanClient(api_url, worker_token, transport=None)`提供`claim/heartbeat/prediction/discovery_page/discovery_done/materialized/fail`；`ModelScanWorker(settings, client, worker_id, predictor=None).run_once() -> WorkerOutcome`。新服务`enqueue_model_selection(receipt_id) -> dict`、`finish_materialization(unit_id, worker_id, attempt_count, binding) -> dict`。内部接口见任务5；网络载荷包含unit_id、worker_id、attempt_count，不由客户端指定新的模型/阈值。

- [ ] **1. 写闭环失败测试。** 用假reader/predictor及MockTransport，只在临时spool写文件；检查阴性不会调用analyze_window，阳性无规则触发仍调用完整分析且`skip_empty=False`。

```python
def test_negative_never_materializes(monkeypatch, scan_worker):
    def forbidden(*args, **kwargs):
        raise AssertionError("negative scan must not create artifacts")
    monkeypatch.setattr("das_sentinel.whole_window.analyze_window", forbidden)
    scan_worker.predictor.reply = {"score": 0.01, "threshold": 0.04,
                                   "label": "absent", "model_id": "fixed-model"}
    scan_worker.run_once()
    assert scan_worker.client.receipts[0]["label"] == "absent"
```

`scan_worker` fixture在本测试文件定义：用任务1服务生成一个infer单元，MockTransport回放任务5接口；reader返回任务3合成数组；fake predictor保存reply；fake client的receipts仅记录成功提交，不使用真实API或模型。
- [ ] **2. 实现先提交选中、后加工。** 每窗先落`negative`或`selected`回执；阳性同事务生成幂等materialize单元，阴性无EventWindow/WindowAnalysis。`prediction`接口逐窗接收成功分数或失败error_code/error_message；失败禁止附带label/score，记unjudgeable后仍可处理同块其余窗口。最后一个回执提交时，由服务逐项核对该infer块全部计划窗的推理结果/排除原因并同事务标单元完成；不能只更新last_index。进程崩溃后从回执缺项恢复。materialize失败保留选中记录与具体错误，可重试加工而不重读整块。
- [ ] **3. 实现阳性入队保护。** 为`enqueue_windows`增加仅供内部已验证receipt使用的独立入口/参数，绕过旧WindowSkip门禁，但不删除skip。事务中先检查431排除、已取消记录、有效人工review及latest_analysis；人工review存在返回`protected`并不改变指针。已有有效图件且输入可验证一致时复用。新建analysis保留真实candidate，skip_policy_version为空；model materialize领取时原子绑定analysis worker/attempt。普通`WindowService.claim`排除由未结束模型materialize单元管理的analysis，防止两个worker重复作图；模型worker15秒心跳同时续租二者。`WindowService.job()`给这些任务返回model_scan_id；普通cancel/retry接口对此返回409 `use_model_scan_control`，操作由批次控制统一处理，不能留下互相矛盾的租约状态。
- [ ] **4. 实现严格提交。** 复用`WindowService.result`完整图件/来源校验；耗时数组比较在写锁外完成，最终事务重查receipt/analysis所有权、源身份、窗口人工review、取消状态和比较摘要，才将receipt标ready并挂不可变模型判定。比较窗对应的来源身份与读取覆盖，不要求整个连续块的多余上下文与单窗完全相同。最终图件波形标准化后，若model_input_sha256与初筛相同可复用初筛分数；若仅在已验证浮点容差内不同，必须用冻结模型对最终数组重算，分别保存screening_score和final_score，复核展示后者及最终input绑定。超出容差或重算跨过阈值均记input_mismatch待处理，不发布。若窗口在推理期间被人工复核，以现有review为准；不改其analysis指针。
- [ ] **5. 补关键测试。** 旧skip+新阳性成功；已有review的latest_analysis和review哈希前后不变；在“比较完成→提交”之间插入人工review仍保护；取消时迟到结果409；同一回执重放不重复analysis；source变化/网格掩码不一致不发布；子进程崩溃重启不会把未知结果写absent；错误窗口计入unjudgeable。
- [ ] **6. 验证。** 主环境运行`tests/test_model_scan_worker.py tests/test_model_scan_materialization.py tests/test_window_worker.py tests/test_window_worker_api.py tests/test_window_skips.py tests/test_event_windows.py`。不启动线上worker。

## Task 5：API、统计及独立模型推荐

**Files:** 新增`api/model_scan_schemas.py`、`api/routes/model_scans.py`、`tests/test_model_scan_api.py`；修改`api/app.py`、`api/routes/event_windows.py`、`services/event_windows.py`；扩展`tests/test_event_windows.py`。

**Interfaces:** 外部URL带`/api/v1`：`POST /sites/{site_id}/model-scans`；`GET /model-scans`及`GET /model-scans/{id}`；`POST /model-scans/{id}/{pause|resume|cancel|retry}`。内部`POST /internal/model-scans/claim`；`POST /internal/model-scans/{unit_id}/{heartbeat|prediction|discovery-page|discovery-done|materialized|fail}`。新增复核查询参数`model_recommended: bool=False`及可选`model_scan_id`；与旧candidate_only/recommended同时为true返回422。

- [ ] **1. 写API失败测试。** 沿用现有`tests.test_events_api`登录fixture；未登录不可读，缺CSRF不可创建/控制，错误worker token不可领任务；合法领取仅本模式，不能领旧任务。注册路由前断言404，再实现后检查200/201与拒绝状态。
- [ ] **2. 固定创建载荷。** 用户发送request_id、directories、include_rule_skips=true，可附time_ranges限定试扫；服务在事务外读取站点及checked_metadata、冻结deployment，写事务内复查站点/profile/模型指针未变化。TimeRange定义为带时区的start_utc/end_utc，end>start，转换UTC、合并重复区间后冻结；最多16个范围，30秒核心窗完全包含于区间才入队。目录白名单和431排除清单由服务器读取并固定SHA，不接受HTTP任意排除清单路径。保留本次期望model_id，若已换模型返回409，不能默默用新模型。

```python
class TimeRange(BaseModel):
    start_utc: AwareDatetime
    end_utc: AwareDatetime

class ModelScanCreate(BaseModel):
    request_id: str = Field(min_length=1, max_length=100)
    directories: list[str] = Field(min_length=1, max_length=12)
    include_rule_skips: Literal[True] = True
    expected_model_id: UUID
    time_ranges: list[TimeRange] | None = Field(default=None, min_length=1, max_length=16)
```

- [ ] **3. 复用鉴权与短事务。** 读取用AuthenticatedAdmin、写入用CsrfAdmin，worker用现有WorkerToken；写操作沿用`write_coordinator.lock`、run_in_threadpool、`session_factory.begin()`。NAS、SHA/数组读取和绘图不得放写锁内；API启动不直接扫盘或自动恢复cancelled。请求限制单页来源256条、单窗prediction、所有数组为磁盘已验证路径，不在HTTP传大数组或密钥。
- [ ] **4. 实现模型推荐查询。** 只join有效`ready`回执且analysis_id/input_sha256仍对应当前成功analysis；用EXISTS/单窗选定回执避免多批重复行。指定scan则按该批次，不指定则取该窗口最新ready绑定；按score降序、start_utc、window_id稳定排序。candidate保持原值，模型与规则分数分开。`WindowItem/Detail`增加可空`model_recommendation={scan_id,model_id,model_name,score,threshold,analysis_id,input_sha256}`；已人工复核按已有review_status处理，不改变标签。
- [ ] **5. 实现统计与ETA。** 服务返回`discovery_complete`、sources、planned、judged、selected、negative、unjudgeable、protected、excluded、pending_artifacts、ready；selected/ready为阶段指标不能与negative等相加当总数。每窗只统计一次，复用旧完成回执也单列reused。来源未枚举完成时total/ETA为null；扫描与补图分别用近期至少4个完成样本的真实墙钟吞吐估计，等待worker/暂停时显示状态，不把未知显示0秒。pending_artifacts>0不得显示批次成功；加工失败且没有可运行单元时批次为failed，可retry；只有已完整发现、全部推理/加工终结且存在unjudgeable窗时才为completed_with_errors。
- [ ] **6. 验证。** 测试ready但candidate=false仍列出；低分在前旧优先级不干扰模型排序；换active模型后结果版本/分数仍冻结；分页无重复；错误及未出图不计完成。运行`tests/test_model_scan_api.py tests/test_event_windows.py tests/test_window_inference_api.py tests/test_scan_api.py tests/test_job_estimate.py`。

## Task 6：任务页与人工复核入口

**Files:** 新增`frontend/src/modelScanApi.ts`、`components/ModelScans.tsx`、`components/ModelScans.test.tsx`；修改`pages/JobsPage.tsx`、`components/WindowJobs.tsx`、`windowApi.ts`、`pages/WindowEventsPage.tsx`及对应测试；样式沿用现有任务卡片，必要规则局限于新组件。

**Interfaces:** `modelScanApi.list/detail/create/action`复用现有api cookie/CSRF；批次类型对应任务5。`WindowItem.model_recommendation`与后端同名，不把score解释为概率。

- [ ] **1. 写行为失败测试。** MSW响应一个仍在发现的批次、一个推理完但待出图批次；前者显示“正在统计数据范围”而非精确百分比，后者不显示全部完成。暂停/继续发正确接口，取消后不出现可继续按钮。参考既有WindowJobs测试初始化方式。
- [ ] **2. 新增批次显示。** JobsPage现有WindowJobs旁增加ModelScans：显示模型/目录范围、阶段、已判断/入选/待出图/可复核数量、剩余时长与完成时刻；10秒轮询且卸载AbortController，沿用现有错误/重试样式。列表分页，后台分页返回而非一次装入所有回执。初版创建使用站点下批准目录多选及include_rule_skips固定说明，不增加任意阈值编辑。WindowJobs行若有model_scan_id，提供所属批次入口以替代普通单窗取消/重试，防止对同一加工任务出现两套互相矛盾的控制。
- [ ] **3. 新增模型推荐范围。** WindowEventsPage范围增加`model_recommended`并保持默认review=unreviewed；切换范围重置页码与selected。任务卡的“查看入选窗口”链接传scan_id。模型推荐队列显示“模型得分”，不显示旧“按目录批次优先级”说明；选择窗后显示冻结版本判定，存在model_recommendation时不自动用WindowPrediction重算当前active模型替代它；普通窗口保留原组件行为。

```tsx
<option value="model_recommended">模型推荐</option>
```

```ts
model_recommended: String(query.range === "model_recommended"),
candidate_only: String(query.range === "candidates"),
recommended: String(query.range === "recommended"),
```

- [ ] **4. 验证可复核行为。** 标签保存后队列仍标出人工状态；无局部区域模型阳性可打开原始/滤波图，显示真实“未触发合格信号”，可人工标注。UI不自动提交标签、不把阴性写人工噪声。测试用户看到的版本/图件/状态而非CSS实现。
- [ ] **5. 执行测试与build。** 在frontend目录运行`npm.cmd test -- --run src/components/ModelScans.test.tsx src/pages/JobsPage.test.tsx src/pages/WindowEventsPage.test.tsx src/components/WindowPrediction.test.tsx`，再`npm.cmd run build`。发布前再跑完整前端测试一次。

## Task 7：有界分块读取、等价性与公平提速基准

**Files:** 修改`model_scan_input.py`，必要时修改`readers/continuous_low_frequency.py`；新增`tests/test_model_scan_blocks.py`、`tools/benchmark_model_scan.py`；扩展`tests/test_continuous_low_frequency.py`。

**Interfaces:** `iter_scan_inputs(plans: list[dict], metadata: dict, cache_root: Path, *, read_mode="single") -> Iterator[tuple[dict, dict, dict]]`依序产出plan/arrays/provenance；read_mode仅`single/block`，初始默认single。`--read-mode`只用于worker启动与基准，不做用户阈值/科学协议调节。

- [ ] **1. 写边界失败测试。** 生成300秒连续核心+两侧真实40秒上下文，原生采样率100/1000Hz，跨分钟、带通外高频、尾部短块；取每个30秒主窗与旧路径对照。

```python
for (plan, expected, _), (_, actual, _) in zip(single_inputs, block_inputs, strict=True):
    assert plan["start_utc"].endswith("+00:00")
    for key in ("distance_m", "relative_seconds", "valid_channel_mask",
                "recognition_channel_mask", "region_id"):
        np.testing.assert_array_equal(actual[key], expected[key])
    np.testing.assert_allclose(actual["waveform"], expected["waveform"],
                               rtol=1e-6, atol=1e-7)
```

以上为合成测试预设上限，不可为了让测试过而放宽。实测还要报告绝对/相对误差分布；任何接近阈值窗的标签差异均阻止block上线，退回single。
- [ ] **2. 实现有界复用。** 将多个30秒窗组合，真实上下文与旧20秒core/10秒halo及UTC采样相位保持一致；保护区不同导致浮点误差时优先复用已验证20秒块，而非换重采样算法。缓存仅针对仍未变化的源身份，使用bounded LRU，最多一个预取；原始读取仍≤128MiB缓存、单计算组≤2百万原生数值、输出<256MiB。源before/after检查不能因缓存删除；一个窗口不合格不吞掉同块其余有效窗口，退回逐窗诊断。
- [ ] **3. 记录基准。** 脚本接受固定source清单、选定窗起点清单、model_snapshot、output及read_mode，调用同一冻结模型。输出JSONL逐窗阶段耗时/错误/score，汇总window/minute、NAS字节读取（能测则记录，不能测则注明）、内存峰值、spool峰值、阳性比例。基准旧全流程、single模型优先、block模型优先在相同窗口/线程分别运行，冷暖缓存分组；不能把历史0.2018秒和83.117秒相除。
- [ ] **4. 审查阈值附近与真实数据。** 先在已有已标窗取正负各10个、优先最近已复核误判窗，另选3种原生网格各至少1段连续5分钟；若某类确无可用源，在报告写明且该类继续single路径。从这些实际分数中取距离阈值最近的10个复算，要求score绝对误差≤1e-6、分类一致；全体网格/掩码精确一致。所有抽样是只读验证，不改标签、不重新训练。
- [ ] **5. 验证与选择。** 运行`tests/test_model_scan_blocks.py tests/test_continuous_low_frequency.py tests/test_model_scan_input.py`，Torch环境跑对应分数对照。满足一致性、无持续内存增长且同数据端到端吞吐优于single才选择block；否则保持single常驻推理与阳性出图，仍记录已经测得的改善及瓶颈。

## Task 8：整体审查、备份、发布、小批试扫和分目录推进

**Files:** 新增`docs/model-guided-scanning.md`；更新`docs/operations.md`；执行记录位于`D:/das-sentinel-runtime/manual-work/model-guided-scan-20260925`。本任务在前7项通过后执行。

- [ ] **1. 审查最终差异与回归。** 独立审查者只审本功能及既有接口影响，重点五类Review Focus；修复后只重复受影响检查。主环境执行本计划所有新测试及旧window/scan/inference/reader相关回归；Torch环境明确运行推理等价测试；前端`npm.cmd test -- --run`与`npm.cmd run build`。保存实际命令、退出码和摘要，不能以以前259项通过代替本次验证。
- [ ] **2. 保存上线基线。** 读取当前Git分支、差异；用SQLite backup API生成一致性备份，不直接复制活动WAL数据库。记录WindowReview、WindowLabelImport、EventWindow及latest_analysis绑定、模型部署文件SHA、431排除清单SHA；哈希基线不把用户期间正常新标签当破坏，按既有review ID逐条比对。

```python
import sqlite3
from pathlib import Path

source = "file:D:/das-sentinel-runtime/app-data/das-sentinel.sqlite3?mode=ro"
target = Path("D:/das-sentinel-runtime/manual-work/model-guided-scan-20260925/before.sqlite3")
target.parent.mkdir(parents=True, exist_ok=True)
if target.exists():
    raise FileExistsError(target)
with sqlite3.connect(source, uri=True) as src, sqlite3.connect(target) as dst:
    src.backup(dst)
```

- [ ] **3. 先在备份副本迁移。** 验证新表、旧行/标签一致、foreign_key_check/integrity_check通过；再短时停止当前API、对线上库应用新增迁移、使用已有配置重启API。不修改秘密、不重新初始化管理员；新服务失败时停止新worker并恢复旧API可执行版本，保留新增表/回执，不用旧整库备份覆盖用户可能新增的复核。
- [ ] **4. 发布与检查。** 静态资源当前实际目录为`D:/das-sentinel-runtime/manual-work/window-images-recovery-20260925/frontend-dist`，发布前再从API配置核对。沿用追加新哈希assets、备份并原子替换index；验证healthz、登录/session、既有标签页、模型扫描列表及模型推荐页。浏览器另开验证页，不刷新正在编辑标签的用户页。
- [ ] **5. 小批试扫。** 使用管理员API通过directories和time_ranges创建覆盖三种可用原生网格及新日期的限定计划，不放宽HTTP路径限制。包含至少1个已知阳性、1个旧规则skip（若其来源仍可读）、1个已删排除身份作不入队断言；先核对数量，不把无法定位的目录随意替换为全盘。已复核阳性只作保护/推理对照，不重复生成其人工标签或绑定。隐藏启动一个模型scan worker，日志与PID记入操作目录；运行中核对API仍可复核、失败分类、回执重放、暂停后继续、阳性图件、无STA阳性保留、阴性无事件/图件。阴性只做只读抽查，不赋人工标签。
- [ ] **6. 试扫验收后批量。** 对12个白名单目录逐批提交，include_rule_skips=true，模型/阈值/排除清单固定；用持久来源与回执跨批去重。不要恢复旧cancelled任务。目录登记规模和可读覆盖明确后才给剩余总量/ETA；不承诺全部NAS数据本轮对话内完成。处理中的扫描留在可恢复后台，报告批次ID、真实进度、日志和停止/继续入口。
- [ ] **7. 最终核验与文档。** 逐条确认基线人工review及旧图件绑定未变、431排除窗仍未重建、active.json及权重SHA未改。报告实际吞吐、试扫模型入选比例及局限；连续背景误报率仍需用户复核，模型得分不是校准概率。记录全部测试及部署结果，不将待扫进度写成扫描完成。

## 自审覆盖与执行交接

- 设计的模型/输入固定：任务1、3、5；规则skip与人工保护：任务2、4；持久去重/恢复：任务1、2、4；只阳性加工：任务3、4；模型推荐/ETA：任务5、6；实际提速与回退：任务7；上线和逐目录推进：任务8。
- 执行顺序为1→2→3→4→5→6→7→8。接口依赖紧密，推荐由主代理在当前任务内逐项实施，辅以独立只读审查；也可按任务委派实现并逐项审查。计划自审由主代理完成，不另委派计划自审。
- 用户已批准设计、计划和当前任务内实施；主代理按 executing-plans 执行，独立审查已完成。上线试扫后继续分目录提交，不重复请求已获授权的确认。
