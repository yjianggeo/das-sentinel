# 30 秒窗口规则检测与双图复核实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox syntax.

**Goal:** 让 30 秒全通道训练窗口提供传统规则筛选、双瀑布图编号框和局部预览。
**Architecture:** 不改变既有数值和检测算法，新增衍生图件版本、区域摘要和受保护的缓存数组预览接口；前端消费固定接口，既有标签按原签名保留。
**Tech Stack:** Python/NumPy/SciPy/Matplotlib/FastAPI/SQLAlchemy；React/TypeScript/Vitest。
**Spec:** `docs/superpowers/specs/2026-09-16-window-signal-review-design.md`

## Global Constraints

30 秒、100 Hz、全原生通道和站点识别掩码不变；无 AI 自动判类；局部框不成为训练样本；不删除数据、不恢复取消任务、不提交/推送/合并；沿用 fix/jjk-scan-progress 工作树及已有中文修改。

### Task 1: 规则证据区域与双瀑布图（root）

Files: whole_window.py、whole_window_plots.py、新 window_regions.py；tests/test_window_regions.py、test_whole_window.py。
- [x] 先写并运行失败测试：qualified 合并/边界/弱证据、两类图件且原数组保留。
- [x] 实现区域合并及固定图形几何、双瀑布图与滤波数组缓存；保留原数值签名。
- [x] 跑聚焦数值/图件回归，自查差异。

### Task 2: 受保护的筛选和局部预览 API（root）

Files: services/event_windows.py、api/routes/event_windows.py；tests/test_window_visual_api.py。
- [x] 先验证新查询、manifest、区域预览测试失败。
- [x] 实现 spec 固定接口；旧结果兼容，过滤后分页，局部预览只读取已校验数组。
- [x] 验证外窗分析/非法区域/损坏文件拒绝、局部至少3秒、原整窗标签回归。

### Task 3: 复核 UI（限定文件独立子代理）

Files: frontend/src/pages/WindowEventsPage.tsx、WindowEventsPage.test.tsx、windowApi.ts、components/WindowSignalViewer.tsx 及测试、window-events.css（仅追加 signal/viewer 样式）。
- [x] 先测试 candidate_only 默认查询、双图无波形、点击框/列表请求绑定 analysis_id 的局部图。
- [x] 新组件接收 detail；以 spec detection_regions/plot_geometry 生成图内可点击框及有限高列表，保留下载和全屏键盘操作、错误重试和 Blob 清理。
- [x] 保留整窗多标签和旧线索，替换机器类别说明为规则检测线索，不调用旧 ROI preview。
- [x] 聚焦前端回归、构建并提交独立审查（不做 Git 提交）。

### Task 4: 整体审查、真实补图及本地部署（root）

Files: docs/whole-window-events.md、本计划 ledger；manual-work/window-signal-review-20260916 操作目录。
- [x] 独立审查 Task1/2/3 并修复，运行受影响后端/前端测试及构建。
- [x] 备份旧代码、数据库及静态入口。真实窗口衍生图件核验原始数值和输入签名不变。
- [x] 补齐已有成功窗口新图，条件更新 result_json 并证明原标签/输入/状态未变；运行报告保留失败及重试依据。
- [x] 切换 API、单 worker 和前端，验证会话、候选筛选、双图、点选局部、任务连续执行。
- [x] 文档及 ledger 记录测试和生产实测范围、已完成补图数，保留旧图/备份与既有 dirty。
