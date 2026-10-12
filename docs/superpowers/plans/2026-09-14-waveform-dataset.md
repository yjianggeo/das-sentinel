# 站点统一波形样本实施计划

Spec: `docs/superpowers/specs/2026-09-14-waveform-dataset-design.md`

## Global Constraints

使用既有 `fix/jjk-scan-progress` 工作树，保留全部原有未提交文件；不提交、推送、合并，不操作 worker/API，不修改生产数据库或训练流程。仅写新增离线输出目录。数值数组为 distance × time，全真实通道；标签只覆盖人工复核目标。默认常规 60 秒/100 Hz，低频 600 秒/5 Hz，低频前后各 300 秒滤波上下文另加 10 秒抗混叠上下文。先全量分组再选择导出目标。

## Task 1: 连续读取器支持两种输出采样率

文件：`readers/continuous_low_frequency.py`、`tests/test_continuous_low_frequency.py`。

先添加 100 Hz UTC 网格、抗混叠、非法率与间隙测试并运行失败。为现有函数添加 keyword-only `sample_rate_hz: int = 5`，仅支持 5 和 100；替换局部网格/分块计算，保留默认行为和全部安全检查。100 Hz 接缝最多允许原始采样间隔（且不超过既有 0.05 秒），防止高频严重插值。原 5 Hz 调用与测试不变。

## Task 2: 只读事件清单和分组

新增 `services/waveform_manifest.py`、`tests/test_waveform_manifest.py`。

接口 `create_manifest(database: Path, site_id: str) -> dict`：SQLite mode=ro、query_only、一致性 BEGIN；元数据缓存按源路径读取；取最新 Review，保存 source/grid/time/ROI/review/legacy_features。全部常规已复核事件，人工 unknown 不作四类样本。目标字段使用 `event_id, review_id, label, reviewed_at, source_path, source_fingerprint, start_utc, end_utc, channel_start, channel_stop, group, legacy_features`。清单包含 `schema_version, site_id, created_at, targets, excluded`。根据最长读区间合并重叠 UTC 日组。元数据失败进入 excluded，不能中止其他目标或伪造时长。测试先行。

## Task 3: 样本构建、存储、加载

新增 `services/waveform_samples.py`、`tests/test_waveform_samples.py`。

接口 `build_sample(paths, start_utc, view, reference_distances, reference_unit, *, suspended_end_m, borehole_start_m, annotations=()) -> (arrays, metadata)`；`view` 为 standard 或 low_frequency。数组包括 waveform、distance_m、relative_seconds、valid_channel_mask、region_id、recognition_channel_mask；低频另有 filtered_waveform。annotations 原样来源但映射成样本 ROI，不把其他区域标噪声。全通道调用 Task 1；固定形状、时间/单位/距离检查，源 provenance；低频复用既有分析滤波实现。保存/加载函数采用 NPZ allow_pickle=False 和 JSON，验证契约；现有路径拒绝覆盖。

## Task 4: 离线命令与小批实测

新增 `waveform_dataset.py`、`tests/test_waveform_dataset_cli.py`、`docs/waveform-datasets.md`。

`prepare --database --site-id --output` 创建完整清单。
`build --manifest --source-dir (可重复) --reference-source --output --view --limit/--event-id --suspended-end-m --borehole-start-m` 从完整清单选择目标，按视图导出，保存成功失败报告并展示进度；每个目标保留同窗已复核标注与全量 group。
`window --source-dir --reference-source --output --view --start-utc --suspended-end-m --borehole-start-m` 导出无标签验证窗口。
目录非递归，每目录上限 2000 HDF5；一次构建缓存元数据候选时间索引，传递只与请求上下文相交的文件，避免每目标打开整日 1440 文件。不盲信文件名。

用全部测试、类型/静态检查、独立审查及真实 JJK 小批验证；记录生产数据及服务未更改、原有文件保留。产物不是已训练模型，明确后续模型对比入口。

执行状态：Tasks 1–4 已完成。最终全量后端1257通过、4跳过、1按默认性能标记排除；独立审查通过。实际数据集含739条复核清单、8个常规及2个低频样本。详细证据见 `2026-09-14-waveform-dataset-progress.md`。
