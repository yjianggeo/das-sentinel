# DAS Sentinel 验收报告

日期：2026-08-02　执行环境：Windows 工作站（本机），无 Docker 守护进程。

## 测试证据

- Python 单元/集成测试：`python -m pytest -q` 全绿（含持久化、检测、融合、分类、
  worker、API、CLI、Compose 静态断言、故障注入）。
- 前端：`npm test -- --run` 11 个用例全过；`npm run build`（tsc 严格模式）通过。
- 故障注入：`tests/test_fault_injection.py` 在真实 uvicorn 线程 + 真实 worker
  下验证——已入队文件被篡改后以 `invalid_claim` 失败，后续文件正常成功，队列
  不阻塞。
- Playwright 浏览器验收：`e2e/` 下 Chromium 用例**已在本机执行并通过**
  （登录 → 事件复核 → 切换瀑布图 → 提交人工标签 earthquake → 已保存 → 模型页）。
  播种栈由 `python e2e/seed_stack.py` 提供（合成 STA/LTA 事件，真实 uvicorn +
  真实 worker 处理）。
- Docker 构建：`compose.yaml / compose.cpu.yaml / compose.gpu.yaml` 与双 Dockerfile
  已交付并通过静态断言测试；本机无 Docker，`docker compose config/build` 未执行。

## 端到端联调中发现并修复的真实缺陷

1. **worker 结果载荷 JSON 序列化**：检测指标含 numpy 标量（int64/float32），
   `submit_result` 直接 `json.dumps` 失败——已在 `_result_payload` 增加
   `_json_safe` 递归转换，附回归测试。
2. **内部结果 schema 过严**：事件特征含 `site_id / cable_type / device_config`
   字符串元数据，schema 限定 `dict[str, float]` 导致 422——已放宽为
   `float | str` 并加长度校验。
3. **SPA catch-all 路由吞掉动态注册路由**：部署态挂载的 `/{full_path:path}`
   使运行期新增 API 路由永远 404（worker 协议测试暴露）——已改为 404 异常
   处理器兜底，api/ 前缀保持 JSON 404。
4. **真实数据几何属性缺失**（见下节）。

## 真实数据只读冒烟（Z:\jjk20240513）

- 文件：`Z:/jjk20240513/2019051809/ZD_3400m_8m_4m_10000Hz_1000Hz_UTC8_201905180928.h5`
  （50 054 248 字节，832 通道，1000 Hz）。
- 发现：该批次真实文件**缺少 `SpatialSamplingInterval` 属性**，原读取器拒绝。
  已扩展读取器：属性缺失时按 ZD 命名从文件名推断道间距（取最后一个 `Xm` 标记，
  本例 4 m），并在元数据警告中标注推断来源。
- 结果：CPU 后端完整分析，墙钟 8.9 s（read 0.45 s / preprocess 4.0 s /
  detect 4.3 s），0 事件，1 条审计候选。
- 完整性：分析前后源文件 SHA-256 均为
  `97c6980f5c5a6a7fa9fffab37b72f23b4f487e6caad60602a909d7cfbf77e79d`，
  大小与 mtime 不变——原始数据只读保证成立。
- 注意：`jjk__3400m_4m_2_...` 批次命名中 `2` 可能为抽取因子，间距推断存在歧义，
  标定该批次前需人工核对几何（见 `docs/site-calibration.md`）。

<!-- DAS_SENTINEL_PERFORMANCE_BENCHMARK:START -->
## Performance benchmark

- Date (UTC): 2026-08-16T02:17:30+00:00
- Source file: `benchmark-data\baseline-1000ch-1000hz-60s.h5` (1000 ch x 1000 Hz x 60 s)
- Requested device: `cpu`

## Versions

- python: 3.12.13
- platform: Windows-10-10.0.19045-SP0
- das-sentinel: 0.1.0
- numpy: 2.5.1
- scipy: 1.18.0
- h5py: 3.16.0
- scikit-learn: 1.9.0

## Runs

### Run 1

- Wall time: 51.82 s
- Peak RSS: 1186.4 MB
- Backend: cpu (fallback: none)
- Event IDs: []
- Events: 0 []

| Stage | ms |
| --- | ---: |
| classify | 0.0 |
| detect | 28029.8 |
| fuse | 0.0 |
| preprocess | 20248.8 |
| read | 3363.5 |
| render | 0.0 |
| total | 51798.2 |

### Run 2

- Wall time: 51.06 s
- Peak RSS: 1187.8 MB
- Backend: cpu (fallback: none)
- Event IDs: []
- Events: 0 []

| Stage | ms |
| --- | ---: |
| classify | 0.0 |
| detect | 28047.2 |
| fuse | 0.0 |
| preprocess | 20145.3 |
| read | 2690.2 |
| render | 0.0 |
| total | 51047.4 |

## Gate summary

```json
{
  "mean_wall_seconds": 51.44,
  "max_peak_rss_bytes": 1187782656,
  "parity": true
}
```

<!-- DAS_SENTINEL_PERFORMANCE_BENCHMARK:END -->

## 准确率声明边界

本报告不包含任何城市/海底站点的准确率声明：无本地标注样本，规则/模型在未标定
站点保持弃权。详见 `docs/site-calibration.md`。
