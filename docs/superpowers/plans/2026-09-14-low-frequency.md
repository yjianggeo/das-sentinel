# 连续低频扫描实施计划

**Goal:** 从真实连续分钟文件发现并复核0.01–0.4Hz候选。
**Architecture:** 独立连续读取、信号分析、持久扫描服务和复核页面，复用现有身份认证与站点数据边界。
**Tech Stack:** NumPy/SciPy/h5py、FastAPI、React、pytest/Vitest。
**Spec:** ../specs/2026-09-14-low-frequency-design.md

不提交或推送Git，保留本次前全部脏改；原始DAS只读，旧事件/标签/任务不重算。

- [x] Task 1: readers/continuous_low_frequency.py + tests/test_continuous_low_frequency.py：先测试拆分、缺口、重叠、漂移、抗混叠，再实现有界读取；输出DasBlock/analysis_start/analysis_stop/sources/warnings。
- [x] Task 2: low_frequency.py + tests/test_low_frequency.py：先测试慢波检出与持续背景抑制，再实现独立滤波、候选、测量；输出filtered_data/valid_channel_mask/candidates/measurements/warnings。
- [x] Task 3: services/low_frequency_scans.py、low_frequency_plots.py、api/routes/low_frequency.py + API/服务测试：先覆盖持久恢复、标签幂等与安全边界，再接异步单队列和图件；app.py注册服务与路由。
- [x] Task 4: LowFrequencyPage.tsx、lowFrequencyApi.ts、导航与样式+测试：起止时间UTC+8、范围、任务状态、图件测量和人工复核。禁止挂载时自动启动扫描或提交标签。
- [x] Task 5: 独立审查、真实地震/背景对照、相关回归及构建；备份后本地上线并核对资源和原数据库，保存进度与结果。

完成记录：[验证与上线](2026-09-14-low-frequency-progress.md)。
