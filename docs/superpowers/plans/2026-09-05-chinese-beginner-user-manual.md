# DAS Sentinel 中文零基础操作手册 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 生成一份零基础用户可以照图操作、并能理解全部 28 个事件特征的中文 Word 手册。

**Architecture:** 以当前运行中的 `http://127.0.0.1:18000` 和项目源码为唯一事实来源。先建立内容核对表并采集无敏感信息的实际界面截图，再用 `python-docx` 生成一份 A4 Word 文档，最后通过 LibreOffice 渲染为逐页 PNG 做视觉验收。

**Tech Stack:** DAS Sentinel React 界面、Python 源码、Codex Browser、python-docx、Pillow、LibreOffice、文档技能渲染工具。

## Global Constraints

- 最终交付格式仅为 `.docx`，文件名为 `docs/DAS-Sentinel-中文零基础操作手册.docx`。
- 文档使用当前访问地址 `http://127.0.0.1:18000`，不能写成默认的 8000 端口。
- 文档覆盖登录、运行总览、站点、任务、事件复核、图件、距离预览、模型、故障处理和管理员附录。
- 事件特征必须覆盖 `backend/src/das_sentinel/features.py` 中的全部 28 个 `FEATURE_NAMES`。
- 规则阈值只能引用 `backend/src/das_sentinel/classifiers/rules.py` 中的当前实现，并标明不是通用物理标准。
- 不记录管理员明文密码、会话信息、密钥或完整文件指纹。
- Z 盘 JJK 原始 HDF5 数据始终描述为只读，禁止指导普通用户移动、覆盖、重命名或删除。
- 当前 18000 服务是本机 Python 服务；Docker Compose 尚无可直接使用的 `.env`，不能声称启动 Docker Desktop 即可恢复服务。
- 最终 Word 必须逐页渲染检查，不能存在截图模糊、表格截断、文字重叠、乱码或孤立标题。

---

### Task 1: 建立事实核对表和完整中文文案

**Files:**
- Read: `README.md`
- Read: `docs/operations.md`
- Read: `frontend/src/pages/*.tsx`
- Read: `frontend/src/components/*.tsx`
- Read: `backend/src/das_sentinel/features.py`
- Read: `backend/src/das_sentinel/classifiers/rules.py`
- Create temporary: `D:/das-sentinel-runtime/manual-work/content.json`

**Interfaces:**
- Consumes: 当前页面名称、操作按钮、状态值、图件类型、28 个特征计算定义和当前分类规则。
- Produces: UTF-8 JSON，顶层键为 `chapters`、`features`、`troubleshooting`；`features` 必须正好包含 28 项，供 Word 生成脚本读取。

- [ ] **Step 1: 核对普通用户可见页面和操作路径**

运行只读检索：

```powershell
rg -n "heading|button|label|tab|机器标签|人工复核|扫描|任务|模型" frontend/src/pages frontend/src/components
```

记录登录、总览、站点、任务、事件复核和模型页面的实际名称、按钮文本和成功状态。

- [ ] **Step 2: 核对事件特征和规则阈值**

运行：

```powershell
Get-Content -Raw backend/src/das_sentinel/features.py
Get-Content -Raw backend/src/das_sentinel/classifiers/rules.py
```

为每个参数写入 `name`、`zh_name`、`unit_or_range`、`plain_meaning`、`high_low`、`possible_relation` 和 `caution`。其中方向参数明确说明 `-1` 与 `+1` 仅表示沿距离坐标的相反方向，`0` 表示未检测到可靠传播方向；`zero_slowness_flag` 明确说明 `1` 是疑似共模或同步噪声标志。

- [ ] **Step 3: 写入快速上手和故障处理文案**

快速流程固定为：打开地址、登录、检查站点、确认数据路径、发起扫描、等待任务、选择事件、查看图件、按需生成距离预览、保存人工复核。每个故障条目固定包含 `symptom`、`likely_causes`、`steps` 和 `escalate_when`。

- [ ] **Step 4: 验证内容核对表**

使用 Python 读取 JSON 并断言：

```python
assert len(content["features"]) == 28
assert {item["name"] for item in content["features"]} == set(FEATURE_NAMES)
assert "20010525" not in json.dumps(content, ensure_ascii=False)
assert "127.0.0.1:18000" in json.dumps(content, ensure_ascii=False)
```

预期：命令退出码为 0，无缺失参数、无明文密码。

### Task 2: 采集并处理当前界面截图

**Files:**
- Create temporary: `D:/das-sentinel-runtime/manual-work/screenshots/*.png`
- Create temporary: `D:/das-sentinel-runtime/manual-work/annotate_screenshots.py`

**Interfaces:**
- Consumes: 当前已登录的本地 DAS Sentinel 页面和 Task 1 的操作顺序。
- Produces: 至少 6 张经过裁切和编号标注的 PNG，覆盖总览、站点、任务、事件详情、300 至 600 米距离预览和模型页面；登录页若无法在不破坏现有会话的前提下采集，则使用无密码内容的实际登录组件截图。

- [ ] **Step 1: 连接当前本地页面并确认无敏感信息**

使用 Browser 控制当前 `http://127.0.0.1:18000`，逐页读取 DOM。截图前确认页面不显示密码、Cookie、CSRF 令牌、密钥或完整指纹。

- [ ] **Step 2: 采集操作截图**

按页面顺序采集全页或局部截图。事件页使用当前 JJK 事件，输入 300 和 600 生成临时预览，并分别采集距离控制区和瀑布图显示区。

- [ ] **Step 3: 添加编号标注和裁切**

使用 Pillow 在副本上添加高对比度编号圆点和简短标签。保持原截图比例，不拉伸；文字区域最小字号在 A4 页面宽度下仍可辨认。

- [ ] **Step 4: 检查截图清晰度和隐私**

逐张打开 PNG，确认编号不遮挡按钮、图件坐标轴完整、没有明文密码和密钥。删除仅用于定位但不进入 Word 的临时截图。

### Task 3: 生成 Word 操作手册

**Files:**
- Create temporary: `D:/das-sentinel-runtime/manual-work/build_manual.py`
- Create: `docs/DAS-Sentinel-中文零基础操作手册.docx`

**Interfaces:**
- Consumes: `content.json` 和经过检查的截图 PNG。
- Produces: 一份 A4 中文 Word 文档，包含标题、开篇说明、目录式导读、编号步骤、截图、28 项参数表、故障处理和管理员附录。

- [ ] **Step 1: 选择文档模板并初始化文档操作**

在未提供视觉模板的情况下打开文档模板选择器；若用户取消或工具不可用，则使用清晰的黑白蓝简洁版式。第一次创建命令前执行且只执行一次：

```powershell
node container_tools/mark_artifact_operation_started.mjs --operation-kind create --expected-output-count 1 --output-format docx
```

- [ ] **Step 2: 编写最小 Word 生成脚本**

脚本使用 `python-docx`，提供以下单用途函数：

```python
def configure_document(document: Document) -> None: ...
def add_step(document: Document, number: int, title: str, body: str, image: Path | None) -> None: ...
def add_feature_table(document: Document, category: str, rows: list[dict[str, str]]) -> None: ...
def add_troubleshooting_item(document: Document, item: dict[str, object]) -> None: ...
```

页面使用 A4 纵向、约 2 厘米页边距；标题为 Word `Title` 样式；正文使用可用中文字体；参数表开启重复表头并允许跨页；图片保持纵横比并限制在正文宽度内。

- [ ] **Step 3: 生成全部章节**

章节顺序必须与设计文档一致。开篇直接告诉读者“日常操作只需浏览器，不需要懂 Python 或数据库”。快速上手使用短句和编号；管理员命令放入单独附录，不穿插在普通用户步骤中。

- [ ] **Step 4: 执行结构检查**

使用 `python-docx` 重新读取成品并断言：标题存在、主要章节标题齐全、表格中 28 个英文参数名各出现一次、`127.0.0.1:18000` 存在、管理员明文密码不存在、所有嵌入图片关系均有效。

### Task 4: 渲染并逐页修正文档

**Files:**
- Read: `docs/DAS-Sentinel-中文零基础操作手册.docx`
- Create temporary: `D:/das-sentinel-runtime/manual-work/rendered/page-*.png`

**Interfaces:**
- Consumes: Task 3 生成的 Word 文档。
- Produces: 完成视觉验收的同一 Word 文档；临时 PNG 仅用于质量检查，不作为最终交付。

- [ ] **Step 1: 使用文档技能渲染器生成逐页 PNG**

```powershell
python render_docx.py "D:/das-sentinel-runtime/waterfall-range-previews-git-20260901-2/docs/DAS-Sentinel-中文零基础操作手册.docx" --output_dir "D:/das-sentinel-runtime/manual-work/rendered"
```

- [ ] **Step 2: 检查每一页**

逐页查看全部 `page-*.png`，检查标题层级、截图清晰度、表格跨页、页边距、中文字体、图注、孤立标题、空白页、重叠和裁切。不能只抽查部分页面。

- [ ] **Step 3: 修正并重新渲染**

每轮只修正已观察到的问题；任何影响版式的修改后重新渲染全部页面并再次逐页检查，直到无可见缺陷。

- [ ] **Step 4: 运行辅助审计**

运行标题和无障碍审计：

```powershell
python scripts/heading_audit.py "docs/DAS-Sentinel-中文零基础操作手册.docx"
python scripts/a11y_audit.py "docs/DAS-Sentinel-中文零基础操作手册.docx"
```

修复缺失标题层级、表头标记和图片替代文本等可执行问题，再进行最终完整渲染。

### Task 5: 最终准确性核对和提交

**Files:**
- Verify: `docs/DAS-Sentinel-中文零基础操作手册.docx`

**Interfaces:**
- Consumes: 最终 Word 文档、源码定义和最后一轮渲染 PNG。
- Produces: 经过事实、隐私和视觉检查的最终交付文件及一个仅包含该文档的 Git 提交。

- [ ] **Step 1: 核对事实和敏感信息**

确认 28 个参数无遗漏；规则阈值与 `rules.py` 一致；事件类型、图件名称、按钮名称和地址与当前软件一致；文档不包含管理员密码、令牌、密钥或可写 Z 盘操作。

- [ ] **Step 2: 验证最终文件可打开**

重新打开 DOCX，确认 ZIP/OOXML 完整；最后一轮渲染页数大于 0，所有页面在 100% 视图下无可见缺陷。

- [ ] **Step 3: 提交最终 Word 文档**

```powershell
git add -- "docs/DAS-Sentinel-中文零基础操作手册.docx"
git diff --cached --check
git commit -m "docs: add Chinese beginner user manual"
```

- [ ] **Step 4: 交付**

在最终答复中仅引用最终 DOCX 一次，说明它包含快速上手、页面图解、全部 28 个参数和故障处理；不链接临时截图、渲染 PNG 或生成脚本。
