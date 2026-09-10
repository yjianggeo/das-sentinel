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

## JJK 目录扫描

任务页按状态和数据目录组织，每页读取 50 条，可用 **上一页 / 下一页** 查看历史。
默认显示全部任务，运行和排队优先，同状态下较新任务在前。状态筛选支持“进行中”
及单一状态；目录筛选使用站点内相对目录，只匹配该目录直属文件。列表显示源文件
名称和目录，切换筛选会回到第一页。所有历史记录仍保留，无需移动或重新分析文件。
顶部状态数量为全部任务的统计，不随当前页或筛选条件缩小；批量删除已取消任务也
作用于所有目录。目录选项在初次加载、手动刷新、扫描完成和删除后更新，自动刷新
只读取当前页。若站点根目录后来变更，旧任务仍保留，并按原始绝对目录显示和筛选。

扫描负责发现文件和创建排队任务，实际读取分析由 worker 执行。若任务一直处于
“排队”、没有“运行中”任务，请先检查 worker 是否已启动并能连接 API；只启动
API 可以登录和扫描，但不会自动分析文件。任务页在存在排队或运行任务时，每次
请求结束后间隔 3 秒刷新，也可点击 **刷新任务**。

进度条颜色与任务状态一致：排队灰色、运行蓝色、成功绿色、失败红色、已取消黄色。
成功任务始终显示 100%，也兼容历史记录中未更新的进度值；后续任务完成时会同步
保存完整进度。其他状态仍显示实际已上报的进度，不提前视作完成。

任务取消后，可使用该行的 **删除**；也可点击 **删除全部已取消任务** 并确认。
删除仅移除已取消的任务记录，保留源文件登记、原始 DAS 文件、已有分析结果、
图件和人工标签。若任务曾关联分析记录，只解除其任务关联。其他状态不可删除。
取消运行中的任务后，worker 在收到取消或任务已删除的响应后会放弃本次结果，
继续领取下一项；当前分析函数可能需要运行到返回才结束，不代表取消状态失效。

1. 打开 **任务 → 选择目录并扫描站点**，选择目标站点。
2. 进入具体的 JJK / 日期目录（例如 `jjk20240608/20240610`），选择当前目录，
   确认页面显示的站点根目录相对路径。必须选择非根子目录，不可省略扫描范围。
3. 只点击一次 **开始扫描**。
4. 等待状态依次经过 **正在扫描 → 请等待稳定性复检 → 正在复检 → 创建任务 → 完成**。
   页面显示文件计数与进度；稳定性等待和第二次检查均由服务端自动完成。
5. 扫描期间按钮禁用，不要重复点击或另行发起第二次扫描。刷新页面后可恢复查看
   仍在运行的扫描。
6. 扫描状态仅保存在 API 进程内存中，API 重启后不能续接原扫描；若因重启或 NAS
   断连中断，请在 API / NAS 恢复连接后重新选择目录并启动新扫描。

原始数据全程只读；扫描仅递归处理选定目录内支持的稳定文件。一个目录含有许多
分钟文件时，每个稳定且受支持的文件会创建或复用一个分析任务，不会把整个目录
合成一个任务。生产目录的验收应只浏览和选择范围，未经确认不要点击开始扫描，
以免批量入队。

CLI 同样必须提供站点根目录相对的 `PATH`，一次命令自动等待两次检查结束：

```bash
das-sentinel scan <site-id> <relative-path>
das-sentinel scan <site-id> jjk20240608/20240610
```

## 新版事件测量与初筛离线比较

事件特征面板可显示 **新版测量** 和 **历史特征**。新任务在检测时自动计算新版测量，历史事件也可通过独立原始数据回放补充，
包括背景信噪比、主要能量持续时间、独立脉冲、固定分析频带下的频谱与空间传播测量。
未取得背景、无法分辨传播或缺少仪器校准时显示原因；“未测得”不能解释为数值为零。
历史特征及已有模型输入保持原样，人工标签不会因测量更新而被重写。

新版测量的主要定义如下，不能直接与归一化后的历史振幅比较：

| 指标 | 定义与使用条件 |
| --- | --- |
| 带通均方根振幅 | 去趋势、固定 0.5–80 Hz 分析带内的 RMS，保留源文件输入单位；实际频带受采样率限制并在测量说明中显示 |
| 信噪比 | `10 × log10(事件窗平均功率 / 事件前背景平均功率)`，事件窗功率包含其噪声；背景取事件前 5 至 0.5 秒，至少需要 1 秒有效背景 |
| 触发持续时间 | 原事件结束与开始样点之差除以采样率；与能量累计 5% 至 95% 的持续时间分别显示 |
| 独立脉冲 | 平滑 Hilbert 包络上的有效脉冲段；平滑尺度 0.1 秒、短间隔合并尺度 0.15 秒，不按载波振荡周期计数 |
| 频谱与频带占比 | 使用真实窗长的频率分辨率；低、中、高频默认分界为 5 与 20 Hz，实际边界在说明中显示，无可解析频点时显示不可用 |
| 传播参数 | 局部通道对的有限时移相关，要求足够非零延迟和一致性；相关峰达到搜索边界或时差不可分辨时不报告速度 |

测量不去共模。事件后使用最多 2 秒滤波保护数据；边界保护不足、上升起点未完整观察、
距离信息无效等情况会说明限制。距离指标沿源文件的光缆坐标解释，不自动视为真垂深。

生产事件检测现在只使用版本2，`DAS_PIPELINE_VERSION` 必须为 `v2`；API 与 worker 必须一起升级并重启。
新任务通过局部多尺度触发与版本2融合生成事件，并自动将新版测量写入注册的证据图件。
历史任务和人工标签保留原版本；旧版失败任务不能重试到新版，需要用户重新扫描建立新的版本2任务。
旧检测实现仅留在显式离线比较模块 `evaluation_legacy.py` 中供研究复现，不是生产可选路径。
部署背景与已知候选数量限制见 [版本2上线说明](production-screening-v2.md)。

离线工具读取数据库快照和原始HDF5，不创建任务、不训练或发布模型。运行示例（使用项目已有Python环境）：

```powershell
python scripts/evaluate_screening.py --database D:/das-sentinel-runtime/app-data/das-sentinel.sqlite3 --output D:/das-sentinel-runtime/manual-work/screening-comparison
python scripts/evaluate_screening.py --database D:/das-sentinel-runtime/app-data/das-sentinel.sqlite3 --output D:/das-sentinel-runtime/manual-work/event-measurements --measurements-only --measurement-dir D:/das-sentinel-runtime/app-data/screening-measurements/v2
```

输出包含冻结样本 `snapshot.json`、选择清单 `selection.json`、逐窗口结果 `report.json`
和可直接打开的 `report.html`，并记录测量与检测代码的 SHA-256。复用同一输出目录时使用已冻结的标签快照；新的独立
实验应使用新目录。可用 `--limit` 先小批量检查读取和耗时。读取失败或源文件发生变化
的样本单独报告，不能作为未检出样本计数。

版本2包含局部空间支持、多时间尺度触发、相干多峰及轨迹几何限制；报告提供关闭
单一检测器、整个振幅组（STA/LTA 与能量）、整个空间组（相干与轨迹）的对照，及部分窗口的共模处理开关对照。已有事件和未复核背景都参与比较，
但背景候选须经盲审后才能统计误报；仅凭旧流程筛出的事件不能计算全目录漏检率。
复核API先读取历史旁路测量文件，缺失时读取该事件已经登记、SHA-256匹配的证据JSON。
两条路径均核对事件身份、采样区间、通道范围、源指纹和有限数值，且限制文件大小并拒绝符号链接；
网页请求不扫描原始数据。没有可信新版测量时继续显示历史特征。

配对比较向两版检测器提供相同的完整数据窗，再保留中央区域的候选峰。这是标准化
窗口比较；它不逐字节复刻生产任务的分块过程。局部支持指标不自动给振幅证据增加
第二票；移除空间组后，独立振幅组仍须达到强证据门槛。报告的耗时为每次算法调用的
墙钟耗时之和；多个独立离线进程并行时，总等待时间需另看执行清单中的总墙钟时间。

当前 Windows 本地实例的在线目录是 `frontend/dist/plot-time-window`。发布构建必须
通过 `npm run build -- --outDir <独立暂存目录>` 输出到在线目录之外，再复制新哈希资源
并原子替换在线 `index.html`。不要在该实例直接运行默认输出到 `frontend/dist` 的构建，
因为 Vite 清空默认输出目录时会同时移除嵌套的在线文件。修改前端不需要重启 worker；
新增 API 路由时仅重启 API，并保留现有环境中的登录密钥和数据库位置。

## 事件图件与精确时间轴

事件详情中的 **事件特征** 使用中文名称并标明单位，数值保持原始特征的显示精度。
桌面布局依次为事件队列、图件与复核、事件特征，左右面板顶部对齐，底部与人工标签
表单底部对齐，并随图件和表单高度自动调整；检测证据和规则判断放在图件下方。
窄屏自动纵向排列，两侧面板使用相同的
较短高度。内容较多时可在面板内滚动，标题保持可见；也可用 Tab 聚焦特征面板后
通过方向键或 Page Up / Page Down 查看。

打开一个已有事件后，页面会自动按事件通道生成瀑布图、滤波瀑布图和波形，并默认
显示带本地参考时间及精确相对秒数的新版图，无需先点击生成按钮。生成期间显示
加载状态；切换事件后为当前事件重新生成，不会批量重跑分析或改写历史 PNG。
距离输入框默认显示事件通道对应的自动距离范围；不修改输入框时，点击
**生成临时预览**可重新生成自动范围的图。只要改过任一输入框，页面就会按手动范围
处理；此时起始、结束距离必须同时保留或填写为可用范围内的有效数值，且起始距离
严格小于结束距离，再点击一次生成。

要查看整段光纤，先点击 **全距离范围**，自动填入当前源文件的起止距离，再点击
**生成临时预览**。也可手动缩小距离范围；距离端点对应的通道包含在选择中。
预览最多读取 2,048 个通道，覆盖当前 832 / 1,663 通道文件；时间窗口仍限制为
30,000 个采样点。全范围读取后仍按当前规则剔除坏道，因此页面显示的有效通道数
可能少于源文件通道总数；瀑布图、滤波瀑布图和波形使用相同的有效通道范围。

生成完成后，可在**瀑布图**、**滤波瀑布图**和**波形**三个标签之间切换；一次生成
会得到共用同一个真实数据时间窗口的三张临时图，页面会提示“当前显示临时预览”。
**下载**始终下载当前标签正在显示的图；在上述三个标签中显示临时预览时，下载的
就是对应临时图。PSD、STFT 和事件特征仍使用原分析结果，不随临时预览改变。点击
**查看原始图件**会恢复自动距离输入并显示保留的历史原图，此时旧图可能仍带 UTC
时间轴；不再编辑直接生成时，仍采用自动距离范围。重新打开或切换事件时，默认
显示新版图。若生成失败且尚无新版图，页面会明确提示并提供原始图件；已有完整
预览时，失败不会替换当前预览。

源文件长度允许时，三张临时图至少显示 3 秒真实数据；若事件及原有上下文更长，则
保留更长窗口。后续新分析正式生成的瀑布图、滤波瀑布图和波形也沿用这一同窗规则；
已有事件的历史原图保持原样。
窗口碰到当前源文件的开头或结尾时，会尽量把缺少的时长移到另一侧。若整个源文件
不足 3 秒，图中会标注 `Source shorter than 3 s; showing actual duration`，并只显示
实际可用时长，不补造样本，也不跨文件拼接。图题中的 `Display samples` 和
`duration` 描述本次显示窗口及其真实时长，不是事件持续时间；滤波图中的
`Band-pass` 以及页面显示的滤波频带表示本次滤波瀑布图实际采用的频率范围。

时间轴上方的 `Local reference` 是站点时区（例如 `Asia/Shanghai`）的整秒参考时刻，
下方 `Time relative to reference (s)` 是相对该时刻的精确秒数。例如第一条显示样本
发生在整秒参考时刻之后 0.234 秒，该样本的横坐标就是 `0.234` s；自动刻度会覆盖
这个时间范围，但不保证恰好把 `0.234` 标成一个刻度。相对时间的零点只是上方参考
时刻，不一定是事件开始时刻；事件位置以图中的事件标记为准。

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
curl -fsS http://127.0.0.1:8000/healthz    # {"status":"ok","pipeline_version":"v2"}
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
