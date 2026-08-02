# DAS Sentinel

面向分布式光纤声学传感（DAS）数据的事件检测与复核平台。系统对只读挂载的原始
ZD/PRODML HDF5 采集文件做幂等分析，产出事件、证据图与机器标签，人工复核后可训练
并发布站点级模型，全程支持 CPU 或 NVIDIA GPU 独立部署。

## 功能概览

- **多道检测融合**：STA/LTA 多道符合、多频带能量统计、去偏相干性、车辆轨迹四个
  检测族，融合为带证据的事件记录。
- **三路分类**：规则判断（始终在线）、相似事件检索、监督模型；证据不足时弃权为
  `unknown`，标签空间严格为 `earthquake / blast / traffic / noise / unknown`。
- **复核优先的 Web 界面**：事件队列 + 瀑布图/PSD/STFT + 独立方法证据三栏布局，
  人工复核不覆盖机器标签；模型发布/回滚需显式确认并受验证门禁约束。
- **幂等任务管线**：文件指纹（大小 + mtime + SHA-256）去重，重启自动回收僵尸任务，
  单文件失败不阻塞队列，原始数据永远只读。
- **CLI**：站点管理、扫描/盯梢、任务管理、事件导出、训练、模型生命周期、基准与
  系统诊断（`das-sentinel doctor`）。

## 快速开始（Docker，推荐）

CPU 部署：

```bash
cp .env.example .env   # 填写 DAS_DATA_ROOT 与两个密钥
docker compose -f compose.yaml -f compose.cpu.yaml up -d --build
```

GPU 部署（需要 NVIDIA Container Toolkit）：

```bash
docker compose -f compose.yaml -f compose.gpu.yaml up -d --build
```

API 与 Web 界面位于 `http://localhost:8000`。原始数据以只读方式挂载到 `/data`，
SQLite 与全部产物写入 `/app-data` 卷——详见 `docs/operations.md`。

## 本地开发

```bash
python -m venv .venv && .venv/Scripts/pip install -e ".[dev]"   # Windows
# 或 .venv/bin/pip install -e ".[dev]"                          # Linux
python -m pytest -q
ruff check backend tests
cd frontend && npm ci && npm test -- --run && npm run build
```

本地启动 API（开发态）：

```bash
uvicorn das_sentinel.api.app:app --reload
```

## 文档

- `docs/operations.md` — 备份、重启恢复、失败重试、健康检查、E2E 验收。
- `docs/site-calibration.md` — 新站点标定流程与准确率声明边界。
- `docs/acceptance-report.md` — 验收证据（测试、基准、真实数据只读冒烟）。
