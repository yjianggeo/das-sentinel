# GitHub CI 修复设计

## 目标

修复 PR #1 中重复出现的三个 GitHub Actions 失败，使 Python、CPU 容器、GPU 镜像和前端四个发布门禁全部通过，然后把当前最新版本合并到 GitHub `main`。

## 已确认根因

1. Python 与 CPU 容器作业都调用 `docker/build-pysqlite3-wheel.sh`。脚本只显式安装 `wheel`，而当前 Python 3.12 构建环境没有可供传统 `setup.py` 导入的 `setuptools`，因此以 `ModuleNotFoundError: No module named 'setuptools'` 退出。
2. GPU 作业在 GitHub 托管的 Ubuntu 运行器上展开 `nvidia/cuda:12.8.1-devel-ubuntu24.04` 时耗尽磁盘空间。失败发生在项目安装之前，错误为 `no space left on device`。

## 获批方案

采用最小、可逆的 CI 修复，不改变 DAS 事件检测/分类逻辑或 GPU 运行时契约。容器入口只允许下述经批准的数据库默认值修复。

- 在 SQLite wheel 构建脚本中同时安装 `setuptools` 和 `wheel`，继续使用已有的 SQLite 3.53.3 与 pysqlite3 0.5.4 SHA-256 固定来源。
- 在 `gpu-image-build` 的 Docker 构建之前，仅清理 GitHub 临时运行器中与本项目无关的大型预装工具目录和现有 Docker 缓存，为 CUDA `devel` 镜像释放空间。
- 保留现有 CUDA `devel` 基础镜像，不降级为可能缺少 CuPy 运行时编译组件的 `runtime` 镜像。
- 保留完整 GPU 镜像构建门禁，不跳过或弱化检查。

## CI 首轮复验后的批准补充

首轮远程复验确认 GPU 镜像构建已通过，同时发现两个既有问题。用户批准按方案 A 扩展修复范围：

1. `zd_hdf5.py` 的文件名时钟与原始时钟比较改为两侧都显式使用 UTC-aware `datetime`。只改变时区表示，不改变相等判断、UTC 偏移量或最终事件时间。
2. `entrypoint-api.sh` 在预检和 Alembic 迁移之前导出可覆盖的默认 `DAS_DATABASE_URL=sqlite+pysqlite:////app-data/das-sentinel.sqlite3`。显式传入的 `DAS_DATABASE_URL` 始终优先，默认路径与 `Settings.database_url` 保持一致。

该补充允许修改上述读取器时间表示和容器入口默认值；不授权其他业务逻辑、入口流程或部署配置变更。

## CI 二轮复验后的批准补充

二轮远程复验确认前端与 GPU 镜像构建已通过，同时暴露两个只在 Linux/已安装包环境出现的既有问题。用户批准继续采用方案 A，以最小范围修复：

1. 将 `tests` 声明为普通 Python 包，使 Linux/Pytest 9 能稳定解析现有的 `tests.fixtures` 与 `tests.synthetic` 导入；不修改测试内容或生产代码导入路径。
2. 将 Alembic 的 `migrations` 与 `versions` 目录声明为随 `das_sentinel` 安装的 Python 包，并让 `upgrade_database()` 直接定位 `db.py` 同目录下的迁移资源，不再通过固定父目录层级推断仓库根目录。
3. `upgrade_database()` 使用完全由代码设置的 Alembic `Config`；数据库 URL、迁移版本和 SQLite 安全检查保持不变。容器入口的 `alembic upgrade head` 流程不改。

验收标准是：测试套件在 GitHub Linux 运行器上能够完成收集；构建出的 wheel 包含 Alembic 环境与两个现有迁移版本；从已安装包调用 `das-sentinel admin init` 可以完成迁移并创建管理员。

## 安全边界

磁盘清理只写入 GitHub Actions 的一次性托管运行器，不在用户电脑、NAS、Z 盘或部署服务器上执行。清理命令使用明确的系统工具目录，不使用未解析变量或广泛根目录目标。原始 JJK/HDF5 数据、数据库和应用配置均不受影响。

## 测试设计

先在 `tests/test_compose.py` 增加两个静态契约测试，并验证它们在当前代码上因缺少目标行为而失败：

1. SQLite 构建脚本必须在调用 `setup.py` 前显式安装 `setuptools` 和 `wheel`。
2. GPU CI 作业必须在 Docker 构建前执行受限的磁盘清理，同时仍保留且只执行镜像构建，不添加 GPU 运行命令。

首轮远程复验新增两项 RED 证据：Ruff `DTZ007` 直接证明文件名时间仍为 naive；`tests/test_runtime.py` 新增入口契约测试，证明数据库默认值尚未在迁移前导出。修复后运行对应 JJK 时间测试、入口测试和 Ruff，再进入完整复验。

二轮远程复验提供另外两项 RED 证据：Pytest 收集阶段以 `ModuleNotFoundError: No module named 'tests'` 退出；容器内管理员初始化以 `Path doesn't exist: /usr/local/lib/backend/src/das_sentinel/persistence/migrations` 退出。迁移修复需增加真实 wheel 内容测试和安装布局迁移测试，验证失败由迁移资源缺失或路径推断错误触发，而不是仅检查源文件文本。

实现后依次验证：

- 新增的聚焦测试；
- `tests/test_compose.py`；
- 后端完整 pytest；
- 前端完整测试与生产构建；
- 可用时在本机 Docker 中构建 CPU 镜像；
- 推送后等待 GitHub PR 的全部 CI 检查结束。

## 交付与合并

修复提交推送到 `docs/chinese-beginner-manual`，由 PR #1 重新运行 CI。仅当全部必需检查成功时，使用普通合并提交将 PR 合并到 `main`；不重写历史，不强制推送。合并后核对 GitHub `main` 指向包含最新手册和 CI 修复的提交。
