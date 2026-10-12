# 30 秒整窗与旧标签迁移实施计划

> **For agentic workers:** 使用 superpowers:executing-plans 在当前工作树逐项实施，完成后使用 requesting-code-review 独立审查。

**Goal:** 默认检测、复核和训练统一为 30 秒全通道窗口，同时完成用户已确认的 1062 条旧复核保守转换。

**Architecture:** 30 秒主窗按 UTC 每分钟 00/30 秒对齐，100 Hz、0.5–40 Hz、全原生通道与既存掩码不变。窗口身份增加时长，历史 60 秒窗口、分析和人工复核保留，不复用为 30 秒签名结果。旧局部标签按新的主窗重新映射，仅完整落窗且 ROI 全部处于有效识别范围的正例可继承；其他类别 unknown，来源变更使继承失效。

**Tech Stack:** Python、SQLAlchemy/Alembic/SQLite、NumPy/SciPy、PyTorch、React/TypeScript。

**Spec:** 用户本轮确认保守标签转换，随后明确要求“60s 的窗太长了，修改为30s”；延续 `specs/2026-09-15-whole-window-events-design.md` 的多标签、来源签名、分组隔离和原始记录保留约束。本文件记录对时长的明确修订。

## 全局约束

- 使用现有 `fix/jjk-scan-progress` 工作树，保留既存 dirty 中文文档和代码；不提交、推送、合并 GitHub。
- 默认窗口 `[T,T+30s)`，T 为 UTC 秒数 00 或 30；数组 `[原生通道数,3000]`，不裁剪或拉伸距离。
- 滤波和 STA/LTA 的前后上下文保持 30 秒，主窗缩短不缩短条件化。检测内部末块只接收主窗内峰值，不能把 30–40 秒上下文当事件。
- 旧 60 秒历史独立保留；默认新扫描与训练只用 30 秒，不混合不同窗长评估。原已取消任务不恢复。
- 1062 是历史次数；取同 ROI 最新结论，同窗合并正标签；局部 noise 不转 absent。自动继承注明来源，后续人工整窗保存优先，来源纠正使继承退出训练并进入待复核。
- 60 秒转换已完成代码、副本与独立审查，但未写生产。30 秒调整完成后才做生产迁移。

## Task 1：窗口身份、读取、图件与验证

Files: `whole_window.py`, `whole_window_plots.py`, `services/event_windows.py`, `persistence/window_models.py`, 新 `0005_window_duration.py`, migrations/env.py；测试 `test_whole_window.py`, `test_event_windows.py`, `test_persistence.py`。

- [x] 先测 00:00:45 对齐为 00:00:30、相同起始不同窗长身份不同、旧 60 秒记录含子表时升级完整保留。
- [x] 实现 `window_start(value, window_seconds=30)`、`window_id(site,start,window_seconds=30)`、`plan_windows(sources,window_seconds=30)`、`analyze_window(...,window_seconds=30)`；仅接受 30/60。
- [x] EventWindow 增加 window_seconds（旧记录默认 60）；唯一约束改为 site/start/window_seconds；重建 SQLite 表时处理外键和原不可变触发器，外键检查通过后恢复启用。
- [x] 主数组、相对轴、频带秒级聚合、尾块峰值归属和图坐标均由实际窗长推导。新版本 whole-window-v2；保留历史 v1 的验证/读取。

## Task 2：worker、API、页面与训练协议

Files: `worker/window_main.py`, `api/routes/event_windows.py`, `services/event_windows.py`, `window_training.py`, `classifiers/window_model.py`, `frontend/src/pages/WindowEventsPage.tsx`, `windowApi.ts`。

- [x] 回归 30 秒 API 只接受 3000 点且 :30 开始有效；历史 60 秒签名仍能展示。
- [x] 默认事件队列、扫描与训练导出筛选 30 秒，历史 60 秒显式入口保留；页面校验按所选窗长确定样本数。
- [x] worker 按领取时长执行；不把上下文、HTTP timeout 或 lease 的 60 秒误改为 30。
- [x] 模型动态时间池化覆盖全部 3000 点，模型包绑定时长；禁止 30/60 混训，误报每小时分母按实际窗长计算。

## Task 3：真实数据重映射与标签迁移

Files: `services/window_label_imports.py`, 迁移操作目录 `manual-work/thirty-second-rollout-20260915`。

- [x] 基于已验证源元数据与 2066 个旧主窗范围，生成不额外扩范围的 30 秒计划，重新计算旧 ROI 交集和 1062 条历史的逐窗来源。
- [x] 在副本中创建 30 秒队列、保守标签转换建议，证明旧表/历史 60 秒 review 不变、重复执行幂等；实际读取/全分析至少一个 30 秒窗口并通过 API 数组/图件/来源签名校验。
- [x] 备份生产 DB 和本轮运行状态，先完成新版 API/前端准备，再切换新 worker，旧 60 秒排队由新计划接管且不重复执行；不恢复原已取消任务。
- [x] 旧标签转换建议持久化；已成功分析者绑定签名，排队完成后自动校验绑定；缺数据/跨窗/排除道仅保留来源线索。

## Task 4：收尾

- [x] 独立审查数值末块/签名/数据库重建/标签继承；修复重要问题后重跑受影响回归。
- [x] 验证真实 API、浏览器 30 秒图轴与原登录、单 worker 连续成功；记录映射/继承/待分析/缺数据/待人工确认的实际数量。
- [x] 更新用户文档和进度，明确已迁移标签不等同全部可训练，也不声明未完成的新模型精度。
