# 站点统一二维波形样本

现在可以从已复核事件构建独立的数值数据集，为站点模型训练和预训练模型比较提供相同输入。生产界面中的“训练模型”仍运行 25 个数值特征的随机森林；此离线工具不会启动训练、发布模型或改写人工标签。

## 两种输入视图

| 视图 | 有效时窗 / 采样率 | 数组 | 用途 |
|---|---|---|---|
| `standard` | 60 秒 / 100 Hz | `waveform[真实通道数, 6000]` | 常规事件的时空结构，保留重采样数值 |
| `low_frequency` | 600 秒 / 5 Hz | `waveform[真实通道数, 3000]` 与 `filtered_waveform` | 长周期背景和 0.01–0.4 Hz 信号 |

100 Hz 输入无法表达 50 Hz 以上信号，因此这是第一版统一输入规格，不能称为全部原始频带。重采样先抗混叠，不直接隔点抽取。常规视图不预先扣空间共模、不做逐通道幅值归一化。低频滤波沿用低频复核算法，前后各读 300 秒真实保护区，并额外读 10 秒抗混叠上下文；不在单个一分钟文件上尝试计算 0.01 Hz。

JJK 2026-06-16 的实际文件有 832 个通道，坐标为 0–3324 m、间距 4 m。导出形状因此为 `832×6000` 或 `832×3000`，不会拉伸到名义总长 3403 m。以一个真实文件固定本数据集的通道坐标和单位，旧年份文件若网格不同会报告失败，应另建数据集。扫描距离范围不限制导出；被扫描排除的悬空段仍在数组中。

## 数组、标签和掩码

每个样本目录包含 `arrays.npz` 和 `metadata.json`，NPZ 以数值数组保存，读取时禁用 pickle：

- `waveform`：float32，距离 × 时间；全通道、真实重采样值。
- `distance_m`、`relative_seconds`：真实距离和相对于 `metadata.start_utc` 的时间。
- `valid_channel_mask`：通道有效性，平直通道为假。原始非有限值或缺少时间上下文会使整个样本失败，不补成有效噪声。
- `region_id`：0 悬空、1 地表、2 井下。JJK 参数为 299 m 和 499 m，边界按真实坐标选择；0–296 m 为悬空通道、300–496 m 为地表、500 m 起为井下。
- `recognition_channel_mask`：有效且不在悬空段的通道。训练适配器必须显式应用，直接把完整数组送进网络仍会看到悬空噪声。地表信号保留，后续可依据实测比较其贡献。
- `filtered_waveform`：只在低频视图提供；0.01–0.4 Hz，滤波无效通道置零并同时标为无效。

人工标签是**事件局部标注**：JSON 保存 review ID、事件 ID、人工类别、原始时间/通道边界及映射后的半开 ROI。目标事件之外的区域仍未标注，同窗可能存在其他类别；不能把整窗强制归为一个类，也不能把其余区域自动当作噪声。每条导出标注都要与实际读取的源文件身份及读取区间相符。目标降采样后没有标注点、或目标范围没有可用识别通道时，监督样本导出失败。常规数据库中人工 `unknown` 不作为四类训练标签。独立低频扫描的 reviews.json 暂未合入这份数据库清单；无标签窗口明确标为 `supervised_eligible: false`。

数据加载接口：

```python
from pathlib import Path
from das_sentinel.services.waveform_samples import load_sample

arrays, metadata = load_sample(Path(r"D:\你的数据集\事件ID"))
x = arrays["waveform"]                       # [distance, time]
use_channels = arrays["recognition_channel_mask"]
annotations = metadata["annotations"]        # 仅这些 ROI 有人工类别
group = metadata.get("group")               # 按组划分，禁止逐样本随机拆分
```

## 命令示例

工具只读 SQLite 和 HDF5，只写新输出目录，不依赖 API 登录或 worker。命令适用于当前工作树和独立 Python 环境。目录不递归，每个源目录最多 2000 个 HDF5；跨日窗口需重复 `--source-dir` 提供相邻目录。索引按元数据读取实际起止时间，不凭文件名推断完整覆盖。

```powershell
$repoRoot = 'D:\das-sentinel-runtime\waterfall-range-previews-git-20260901-2\.worktrees\jjk-scan-progress'
$pythonExe = 'D:\das-sentinel-runtime\.venv\Scripts\python.exe'
$env:PYTHONPATH = "$repoRoot\backend\src"
$env:PYTHONIOENCODING = 'utf-8'
$datasetRoot = 'D:\das-sentinel-runtime\training-datasets\jjk-waveform-20260914'

# 全部最新人工复核标签先做快照和分组；新版本请选择新的输出目录。
& $pythonExe -m das_sentinel.waveform_dataset prepare `
  --database 'D:\das-sentinel-runtime\app-data\das-sentinel.sqlite3' `
  --site-id 'cce64999-eb01-4b47-8544-1301ebd6b33c' `
  --output "$datasetRoot\manifest"

$sourceDir = '\\169.254.9.66\Public\jjk2025_2026\20260616'
$referenceSource = Get-ChildItem -LiteralPath $sourceDir -Filter '*.h5' |
  Sort-Object Name | Select-Object -First 1 -ExpandProperty FullName

# 从完整清单选一个已复核事件；将事件ID替换为清单中的真实值。
& $pythonExe -m das_sentinel.waveform_dataset build `
  --manifest "$datasetRoot\manifest\manifest.json" `
  --source-dir $sourceDir --reference-source $referenceSource `
  --output "$datasetRoot\standard-example" --view standard `
  --event-id '清单中的事件ID' --suspended-end-m 299 --borehole-start-m 499

# 北京时间 2026-06-16 17:05 的十分钟低频窗口，仅验证信号，不生成类别真值。
& $pythonExe -m das_sentinel.waveform_dataset window `
  --source-dir $sourceDir --reference-source $referenceSource `
  --output "$datasetRoot\low-frequency-example" --view low_frequency `
  --start-utc '2026-06-16T17:05:00+08:00' `
  --suspended-end-m 299 --borehole-start-m 499
```

`build` 不指定事件 ID 时默认只尝试清单前 20 条，可用 `--limit` 改变小批数量。清单中其他目录的目标如果没有提供相应源目录，会明确失败。含失败时命令退出码为 1，已成功样本保留；每批 `report.json` 列出成功/失败、原因、形状、耗时和源索引问题。所有输出目录拒绝覆盖，便于保留可复现版本。

## 公平比较模型

准备清单时先处理全部已复核事件，按站点 UTC 日分组，并合并最长视图读取上下文（含原始插值邻点）重叠的日期组。再按事件 ID 或 limit 导出，仍保留全量 group；常规、低频视图共享 group。训练集、验证集、测试集必须按 group 划分，归一化统计只拟合训练集。日期组不是已核实的地震事件 ID，后续若完成目录关联，应进一步把同一地震的全部记录合组。

这层数据已保留旧 `legacy_features`，便于对同一成功导出的事件集合比较原随机森林基线和波形模型。首先考虑冻结预训练编码器、训练小型分类头；有足够独立事件后再决定是否微调。DASNet 是首个离线评估候选，需按其官方输入和权重要求做适配，并核实海底数据与 JJK 钻孔数据的差异；不能把该 NPZ 无条件当成任意预训练模型的即插即用输入。低频视图也不能继承只有中高频训练经验的性能结论。

最终比较关注独立地震的召回率、人工复核工作量、各类精确率/召回率和混淆矩阵。数据量、类别均衡与独立日期不足时，应报告证据不足，不以重叠裁片增加的样本数声称训练要求已满足。
