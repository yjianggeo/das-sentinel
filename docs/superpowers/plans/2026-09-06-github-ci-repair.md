# GitHub CI Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复 PR #1 的 SQLite wheel 与 GPU 镜像 CI 失败，在全部发布门禁通过后把最新版本合并到 GitHub `main`。

**Architecture:** 保留现有 SQLite 3.53.3、pysqlite3 0.5.4、CUDA 12.8.1 `devel` 镜像和四项发布门禁。只补齐传统 wheel 构建依赖，并在一次性 GitHub GPU 构建运行器上清理无关的预装工具空间；每个修复先由静态 CI 契约测试失败证明，再做最小实现。

**Tech Stack:** Python 3.12、pytest、POSIX shell、Docker、GitHub Actions、GitHub CLI。

## Global Constraints

- 不修改 DAS 事件检测/分类逻辑、GPU 运行时契约、NAS/Z 盘数据或部署配置；唯一允许的入口变更是为 Alembic 导出与 `Settings.database_url` 一致且可被显式配置覆盖的默认数据库地址。
- SQLite 继续固定为 `sqlite-amalgamation-3530300.zip`，pysqlite3 继续固定为 `0.5.4`，并保留现有 SHA-256 校验。
- GPU 镜像继续使用 `nvidia/cuda:12.8.1-devel-ubuntu24.04`，不得跳过 `gpu-image-build`。
- 磁盘清理只在 GitHub Actions 一次性托管运行器执行，目标必须是明确的无关预装工具目录和 Docker 构建缓存。
- 不强制推送、不重写历史；只有全部必需 CI 检查通过后才合并 PR #1。

---

### Task 1: 补齐 SQLite wheel 的传统构建依赖

**Files:**
- Modify: `tests/test_compose.py`
- Modify: `docker/build-pysqlite3-wheel.sh`

**Interfaces:**
- Consumes: 当前 `pysqlite3-0.5.4/setup.py build_static bdist_wheel` 构建流程。
- Produces: 在调用 `setup.py` 前同时可导入 `setuptools` 和 `wheel` 的确定性构建环境。

- [ ] **Step 1: 写入失败的静态契约测试**

在 `tests/test_compose.py` 增加：

```python
def test_static_sqlite_builder_installs_legacy_build_dependencies() -> None:
    builder = (ROOT / "docker" / "build-pysqlite3-wheel.sh").read_text("utf-8")
    install = "python -m pip install setuptools wheel"
    assert install in builder
    assert builder.index(install) < builder.index("python setup.py build_static bdist_wheel")
```

- [ ] **Step 2: 运行测试并确认 RED**

Run:

```powershell
& 'D:\das-sentinel-runtime\.venv\Scripts\python.exe' -m pytest tests/test_compose.py::test_static_sqlite_builder_installs_legacy_build_dependencies -q
```

Expected: FAIL，因为当前脚本只有 `python -m pip install wheel`。

- [ ] **Step 3: 写入最小实现**

把 `docker/build-pysqlite3-wheel.sh` 中：

```sh
python -m pip install wheel
```

改为：

```sh
python -m pip install setuptools wheel
```

不要改变下载版本、哈希校验或 `setup.py` 命令。

- [ ] **Step 4: 运行聚焦测试并确认 GREEN**

Run:

```powershell
& 'D:\das-sentinel-runtime\.venv\Scripts\python.exe' -m pytest tests/test_compose.py::test_static_sqlite_builder_installs_legacy_build_dependencies tests/test_compose.py::test_cpu_image_pins_verified_static_sqlite -q
```

Expected: 2 passed。

- [ ] **Step 5: 提交任务 1**

```powershell
git add -- tests/test_compose.py docker/build-pysqlite3-wheel.sh
git diff --cached --check
git commit -m "fix: install SQLite wheel build dependencies"
```

### Task 2: 为 CUDA devel 镜像构建释放托管运行器空间

**Files:**
- Modify: `tests/test_compose.py`
- Modify: `.github/workflows/ci.yml`

**Interfaces:**
- Consumes: `gpu-image-build.steps`，其中第 0 步仍为 `actions/checkout@v4`。
- Produces: 第 1 步为受限磁盘清理，第 2 步仍为原 GPU Docker 构建；不得增加 GPU 运行或 smoke 命令。

- [ ] **Step 1: 写入失败的 GPU 空间契约测试**

在 `tests/test_compose.py` 增加：

```python
def test_gpu_ci_frees_hosted_runner_space_before_build() -> None:
    gpu_steps = _ci_workflow()["jobs"]["gpu-image-build"]["steps"]
    assert gpu_steps[1] == {
        "name": "Free space for CUDA development image",
        "run": """sudo rm -rf /usr/share/dotnet /opt/ghc /usr/local/lib/android
docker system prune --all --force
df -h /
""",
    }
    assert gpu_steps[2] == {
        "run": "docker build -f docker/Dockerfile.gpu -t das-sentinel-gpu:ci ."
    }
```

同时更新 `_assert_ci_release_gates()` 的 GPU 步骤精确期望，并在 `test_ci_gate_contract_rejects_missing_steps` 中分别加入 `gpu-space-cleanup` 索引 1 和 `gpu-image-build` 索引 2。

- [ ] **Step 2: 运行测试并确认 RED**

Run:

```powershell
& 'D:\das-sentinel-runtime\.venv\Scripts\python.exe' -m pytest tests/test_compose.py::test_gpu_ci_frees_hosted_runner_space_before_build -q
```

Expected: FAIL，因为当前第 1 步直接执行 Docker 构建。

- [ ] **Step 3: 写入最小工作流实现**

在 `.github/workflows/ci.yml` 的 `gpu-image-build` 中，将清理步骤放在 checkout 与现有 Docker 构建之间：

```yaml
      - name: Free space for CUDA development image
        run: |
          sudo rm -rf /usr/share/dotnet /opt/ghc /usr/local/lib/android
          docker system prune --all --force
          df -h /
      - run: docker build -f docker/Dockerfile.gpu -t das-sentinel-gpu:ci .
```

- [ ] **Step 4: 运行聚焦测试并确认 GREEN**

Run:

```powershell
& 'D:\das-sentinel-runtime\.venv\Scripts\python.exe' -m pytest tests/test_compose.py -q
```

Expected: 全部 `tests/test_compose.py` 测试通过。

- [ ] **Step 5: 提交任务 2**

```powershell
git add -- tests/test_compose.py .github/workflows/ci.yml
git diff --cached --check
git commit -m "ci: free disk space before CUDA image build"
```

### Task 3: 完整验证、推送、等待 CI 并更新 GitHub main

**Files:**
- Verify: `docker/build-pysqlite3-wheel.sh`
- Verify: `.github/workflows/ci.yml`
- Verify: `tests/test_compose.py`

**Interfaces:**
- Consumes: Task 1 与 Task 2 的两个修复提交、PR #1。
- Produces: 全绿的 PR #1，以及包含最新版本的 GitHub `main`。

- [ ] **Step 0A: 复现并修复 JJK 文件名时钟的 Ruff 失败**

先运行：

```powershell
& 'D:\das-sentinel-runtime\.venv\Scripts\python.exe' -m ruff check backend/src/das_sentinel/readers/zd_hdf5.py --select DTZ007
```

Expected RED: `backend/src/das_sentinel/readers/zd_hdf5.py:160:26 DTZ007`。

把 naive 比较改成等价的 UTC-aware 比较：

```python
filename_clock = datetime.strptime(stamp, format_string).replace(tzinfo=UTC)
raw_clock = raw_start.replace(microsecond=0)
```

然后运行：

```powershell
& 'D:\das-sentinel-runtime\.venv\Scripts\python.exe' -m ruff check backend/src/das_sentinel/readers/zd_hdf5.py --select DTZ007
& 'D:\das-sentinel-runtime\.venv\Scripts\python.exe' -m pytest tests/test_zd_hdf5.py::test_interprets_matching_utc8_filename_clock_as_station_local_time tests/test_zd_hdf5.py::test_does_not_apply_filename_offset_when_wall_clocks_disagree -q
```

Expected GREEN: Ruff 退出码 0，两个既有行为测试均通过。

- [ ] **Step 0B: 为容器迁移导出可覆盖的默认数据库地址**

先在 `tests/test_runtime.py` 增加失败测试：

```python
def test_entrypoint_defaults_database_url_before_preflight_and_migration() -> None:
    text = Path("docker/entrypoint-api.sh").read_text(encoding="utf-8")
    export = (
        'export DAS_DATABASE_URL="${DAS_DATABASE_URL:-'
        'sqlite+pysqlite:////app-data/das-sentinel.sqlite3}"'
    )
    assert export in text
    assert text.index(export) < text.index("python -m das_sentinel.runtime")
    assert text.index(export) < text.index("alembic upgrade head")
```

Run:

```powershell
& 'D:\das-sentinel-runtime\.venv\Scripts\python.exe' -m pytest tests/test_runtime.py::test_entrypoint_defaults_database_url_before_preflight_and_migration -q
```

Expected RED: FAIL，因为入口尚未导出该默认值。

在 `docker/entrypoint-api.sh` 的 `set -eu` 后加入：

```sh
export DAS_DATABASE_URL="${DAS_DATABASE_URL:-sqlite+pysqlite:////app-data/das-sentinel.sqlite3}"
```

Run:

```powershell
& 'D:\das-sentinel-runtime\.venv\Scripts\python.exe' -m pytest tests/test_runtime.py -q
```

Expected GREEN: `tests/test_runtime.py` 全部通过。

- [ ] **Step 0C: 提交首轮 CI 暴露的两项根因修复**

```powershell
git add -- backend/src/das_sentinel/readers/zd_hdf5.py docker/entrypoint-api.sh tests/test_runtime.py
git diff --cached --check
git commit -m "fix: make CI runtime defaults explicit"
```

- [ ] **Step 1: 运行完整本地检查**

```powershell
& 'D:\das-sentinel-runtime\.venv\Scripts\python.exe' -m pytest -q
Set-Location frontend
npm test -- --run
npm run build
Set-Location ..
```

Expected: 后端、前端测试和生产构建全部退出码 0。恢复 `npm run build` 产生的 `frontend/tsconfig.tsbuildinfo` 工作树变化，使提交范围保持干净。

- [ ] **Step 2: 在本机或 GitHub CI 复现 CPU 构建路径**

```powershell
docker build -f docker/Dockerfile.cpu -t das-sentinel:ci-repair .
docker run --rm --entrypoint python das-sentinel:ci-repair -c "import pysqlite3.dbapi2 as db; assert db.sqlite_version == '3.53.3'"
```

Expected: 镜像构建成功，容器内 SQLite 版本断言通过。此验证不挂载 NAS/Z 盘数据。

若本机仍因已记录的 `auth.docker.io` DNS 故障无法下载基础镜像，则按用户明确批准，以推送后的 GitHub `cpu-container` 作业作为替代验证；不得因此跳过远程 CPU 门禁。

- [ ] **Step 3: 确认提交范围并推送**

```powershell
git status --short
git log --oneline github/docs/chinese-beginner-manual..HEAD
git push github docs/chinese-beginner-manual
```

Expected: 工作树干净，推送成功且不需要强制更新。

- [ ] **Step 4: 等待 PR #1 全部检查完成**

```powershell
& 'C:\Program Files\GitHub CLI\gh.exe' pr checks 1 --repo yjianggeo/das-sentinel --watch --interval 20
```

Expected: `python`、`frontend`、`cpu-container`、`gpu-image-build` 全部成功；若任何检查失败，停止合并并按失败日志重新进入根因分析。

- [ ] **Step 5: 合并并验证 GitHub main**

```powershell
& 'C:\Program Files\GitHub CLI\gh.exe' pr merge 1 --repo yjianggeo/das-sentinel --merge
git fetch github main
& 'C:\Program Files\GitHub CLI\gh.exe' pr view 1 --repo yjianggeo/das-sentinel --json state,mergedAt,mergeCommit,url
git log -1 --oneline github/main
```

Expected: PR 状态为 `MERGED`，`github/main` 指向包含两个修复提交、中文手册和既有最新功能的合并结果。保留当前工作树，不删除分支或强制清理。
