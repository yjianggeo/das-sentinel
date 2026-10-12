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

任务页显示“模型扫描”批次和“30 秒窗口任务”，已移除历史文件任务列表。模型扫描冻结创建时的站点配置、模型和阈值，以 30 秒为入口；模型阴性只留去重回执，入选窗口生成图件后进入“模型推荐”。目录范围、距离网格排除和批次恢复方式见 [模型直接扫描](model-guided-scanning.md)。

扫描任务排队但没有运行活动时，请检查原有处理服务是否启动并能连接 API。仅启动 API 可以登录和浏览，不会自动读取分析。暂停批次停止新的领取，当前单元允许完成；恢复仅继续该批次，取消任务不会自动恢复。

普通整窗任务按站点、状态和目录筛选、分页查看；列表数量与全部状态统计分别显示。进度条按状态显示，已成功任务显示 100%。删除已取消任务只删除任务记录，不删除原始 DAS 文件或人工标签。

以下 CLI 文件扫描仍供脚本使用，不对应当前网页的模型扫描批次。

CLI 文件扫描必须提供站点根目录相对的 `PATH`，一次命令自动等待两次检查结束：

```bash
das-sentinel scan <site-id> <relative-path>
das-sentinel scan <site-id> jjk20240608/20240610
```

## 事件队列评分排序

事件复核页的队列提供“时间最新”“评分从高到低”“评分从低到高”三种排序，
默认仍为时间最新。检测评分取该事件所有检测证据的最高得分（0–1），不是机器
分类置信度；队列显示三位小数，缺少检测证据时显示“—”。

按评分排序时，无评分事件始终排在末尾；同分事件按发生时间、事件 ID 倒序排列。
排序在服务端对筛选后的全部结果执行，再分页，加载更多会继续沿用排序和筛选条件。
切换排序会重新加载第一页，不改变事件、机器标签或人工复核结果。
API 对应 `GET /api/v1/events?sort=time_desc|score_desc|score_asc`，列表、详情和
相似事件统一返回 `detection_score`，缺少证据时为 `null`。

## 新版事件测量与初筛离线比较

事件特征面板直接显示版本2测量，不再提供“历史特征／新版测量”切换选项。新任务在检测时自动计算测量，历史事件也可通过独立原始数据回放补充，
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

相干分支会识别持续背景：沿候选的同一传播时移对齐波束，将候选窗与紧邻前后各一窗
的功率比较（默认每窗 2 秒）。只有两侧背景完整有效、相干峰突起度低于相干阈值的一半
（默认 0.04），且候选功率不足较安静一侧背景的 1.5 倍时，才标记为持续相干背景。
该类候选保留在初筛结果及诊断中，但不参与事件分组、边界、代表峰或融合投票。
背景不足或功率无效、达到任一突起门槛时仍保留原有检测资格；不能据此保证所有缓慢起始信号都能检出。
机器“噪声”标签与近零慢度本身不作为删除条件。该改动影响后续扫描，已有事件和人工标签不自动重算。

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
网页请求不扫描原始数据。没有可信版本2测量时显示“暂无事件测量”；加载中和请求失败时分别显示对应提示。

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

事件详情中的 **事件特征** 直接展示版本2测量，使用中文名称、对应单位及各指标的显示精度。
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
全距离临时预览不受站点扫描距离限制：即使 JJK 设置从 300 m 起扫描，预览仍包含
源文件中 300 m 之前的通道，便于人工对照被排除的悬空光缆信号。
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

## 站点扫描距离

在**站点**页点击对应站点的**设置扫描距离**，填写起点和终点（单位 m）后保存。
起点或终点留空表示该端不限，两端都留空恢复全距离扫描。例如 JJK 起点填写
`300`、终点留空，即排除原文件光缆距离小于 300 m 的通道。这里使用文件中的
光缆距离坐标，包含边界通道，不把距离自动换算为井深。任务页的扫描入口会显示
该站点当前采用的范围，并提供设置链接。

范围会按站点保存，可随时修改。每次保存创建新的处理配置版本，之后创建的任务
使用新范围；已入队任务沿用原配置，已完成结果和历史标签保持原样。发现文件期间
若修改配置，该次扫描会由配置一致性检查中止，需要重新发起扫描。

范围外通道不参与事件检测、归一化、坏道统计、共模估计、分类特征和事件测量；
结果保留原文件通道编号。若某个文件没有任何通道落入设置范围，该文件任务会明确
报错，不会回退到全距离扫描。原文件不修改，人工复核的临时预览仍可选择全距离。

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

## 全部扫描任务的预计完成时间

任务页的“整窗图件任务预计完成时间”仅统计已创建的整窗图件任务，不受分页和筛选影响；模型扫描的完整进度与预计完成时间显示在独立批次中。图件估算统计最新分析中排队和运行任务，失败、取消、完成及被取代的分析不计入。页面可见且有活动任务时每 5 秒更新，空闲或隐藏时每 60 秒检查，返回页面或操作后立即刷新。

`GET /api/v1/jobs/estimate` 需要登录，只读数据库。整窗队列使用最近 2 小时内最多 31 条完成记录的完成间隔估算吞吐量，至少需要 4 条；使用结果发布前的最后心跳，并纳入空窗口跳过记录的完成时刻。不会把任务排队时间当作处理耗时，吞吐量已包含该队列的并发效果。

样本不足时显示“正在积累估算样本”；超过处理租约时间未收到心跳或完成活动时显示“等待处理服务恢复”。任一有待处理任务的队列无法估算时，总完成时刻保持空白。无待扫描任务时单独提示，不代表失败任务已成功。请求超过 15 秒或发生错误时隐藏旧估算并重试。

这是基于近期速度的近似值，新增任务、NAS 读取速度、任务复杂度及并发变化都会使其调整。此功能不会启动或重启 worker，也不会重试失败或恢复取消任务。

## 30 秒空窗口提前跳过

新 worker 领取未处理的 30 秒任务时启用 `empty-regions-v1`。先按原规则读取、降采样和检测局部区域；没有合格区域时，停止整窗特征测量、训练数组压缩和绘图。服务端在同一事务内记录简短 `window_skips` 去重信息，并删除本次任务、窗口及仅含旧局部噪声的未应用导入草稿。事件列表及训练清单不会新增人工“仅背景”标签。

已有图件、整窗人工复核、历史人工非噪声导入、多版本分析及旧 worker 已开始处理的任务继续原流程。服务端领取时与删除前各检查一次保护条件，同时核验租约、尝试次数、文件和配置。原始文件与历史局部事件/复核不删除。文件或配置未变时重复目录扫描不会重建空窗口；计划签名变化后允许重新分析。

升级需要先备份数据库并执行迁移 `0006_window_skips`，再切换 API 与支持跳过的 worker；旧 worker 协议仍兼容。保留去重表，不可回退时整体覆盖正在增长的数据库。若停用新规则，换回不声明跳过能力的 worker 即可让后续领取恢复完整分析流程；已跳过窗口的重新入队需另行审核去重记录。读取和局部检测仍需执行，所以节省的是检测后的整窗处理及存储成本。
# 模型直接扫描的运行说明

新增模式的目录范围、冻结模型、暂停/恢复、异常处理和受控性能数据见 [模型直接扫描](model-guided-scanning.md)。它使用独立扫描 worker，阴性只存回执，阳性补图后进入“模型推荐”。普通历史任务和人工复核记录保持原流程。

## 当前复核界面与运行总览

复核页以图件和人工标签为主：整窗保留有效通道比例、活动通道比例、连续活动距离和相邻道相关性四项摘要；井下广覆盖、传播及共模残余评估仍独立显示。“更多测量参数”已移除，新分析不再计算这些无其他调用方的辅助整窗输出。模型输入、检测证据与评分保持原协议。

局部疑似区域默认收起，展开后才读取局部预览。新生成的整窗 PNG 不烧录区域框，疑似区域开关控制页面叠加；旧图件沿用兼容预览路径。显示叠加框时下载的仍是原始整窗 PNG。

运行总览统计当前 30 秒窗口的最新分析状态、模型扫描批次状态，以及已完成窗口的人工地震类／噪声类／待复核数量。最新标签须绑定当前分析及输入摘要；缺失、待确定或过期标签均列为待复核，不回退到旧标签。这是当前输入的复核统计，不代表训练可用样本数或模型概率。

本轮验证和性能数据见 [性能与复核界面优化](program-performance-ui-20261010.md)。
