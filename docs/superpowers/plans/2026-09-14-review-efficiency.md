# 复核效率实施计划

**Goal:** 在保留全部数据的前提下，让用户看到训练缺口并优先复核有价值、少重复的候选。
**Architecture:** 两个只读服务及API；模型准备度组件；事件页独立推荐队列组件复用原复核详情。目录窗口使用带来源的本地数据文件。
**Tech Stack:** SQLAlchemy、NumPy/scikit-learn、FastAPI、React、pytest/Vitest。
**Spec:** ../specs/2026-09-14-review-efficiency-design.md

使用原worktree；先失败测试再实现；不修改原标签/模型/扫描、不Git提交推送；在当前会话并行实现独立组件并交叉审查。

- [x] Task 1：backend/src/das_sentinel/services/training_readiness.py、api/routes/models.py、tests/test_training_readiness.py。用最新复核与真实生产同参数split检查计数/schema/UTC日/每折覆盖；测试fit调用即失败、未知标签/空集/不存在站点/混合版本/复核覆盖。完成后提供固定JSON合同。
- [x] Task 2：services/review_batches.py、api/routes/review_batches.py、tests/test_review_batches.py。先用两个站点/跨日/两版本/最新noise更正地震样例验证隔离和推荐，再实现有限特征相似、120秒锚定分组、20成员上限、混合推荐与稳定全池分页。测试缺目录/坏特征/空集、不重复或丢事件、目录线索仅关联相应站点。根代理注册app路由并测试认证API。
- [x] Task 3：frontend/src/reviewApi.ts、components/ReviewBatchQueue.tsx、pages/EventsPage.tsx、pages/EventsPage.test.tsx、新队列测试。先测试推荐模式、站点切换、分页、展开成员及既有复核；实现稳定当前批次、刷新与旧响应隔离。保留普通EventQueue及其筛选行为。
- [x] Task 4：components/TrainingReadiness.tsx及测试、pages/ModelsPage.tsx及测试。先验证真实缺blast两折时显示原因/阻止误触训练、切换站点隔离，后实现计数表和刷新。根代理实现此部分并协调样式。
- [x] Task 5：导入已计算目录窗口，真实数据库只读预览与耗时检查，交叉独立审查，相关前后端回归与独立outDir构建。备份后仅重启API/替换前端，浏览器验证，再比对旧数据库和Git脏文件并写完成记录。
