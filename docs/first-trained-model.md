# JJK 首个已训练模型与预训练迁移比较

2026-09-21 更新：按用户要求，本文的旧 ROI 注册模型和旧离线权重已移出原使用位置，归档到 `D:/das-sentinel-runtime/manual-work/binary-review-cleanup-20260921/archive`；本文保留为实验历史。当前系统采用 30 秒整窗二分类复核，首个整窗候选和清理结果见 `whole-window-events.md`。

截至 2026-09-14，首个 API 随机森林候选和独立迁移比较均已训练完成，均未发布。554 条共同波形样本上的开发集比较选中去掉时钟特征的 `extra24_leaf3`；其一次固定留出测试宏平均 F1 为 **0.588397**，仍不足以支持部署。DASNet 冻结主干加本地四类分类头已训练并保存，但本轮没有证据表明它优于传统特征模型。

## 已核验的 API 候选 v1

站点为 JJK（`cce64999-eb01-4b47-8544-1301ebd6b33c`）。模型 ID 为 `1a796957-268a-4c7c-a73f-f4629c3af675`，版本 `1`，状态 `candidate`，创建时间为 **2026-09-14 07:23:03 UTC／北京时间 15:23:03**。模型使用 25 个数值特征、300 棵随机森林树、类别平衡权重和固定种子 `20260726`；规则配置版本为 `3`，scikit-learn 为 `1.9.0`。

训练快照包含最新人工复核的 **739 个事件 ROI**：地震 40、爆破 35、交通 30、噪声 634。按 UTC 日期分组进行五折验证，每折训练与验证均包含四类，日期不跨折。原始随机分组方案会漏掉爆破类别；覆盖约束只修复分组结构，没有降低标签数量或发布指标门槛。以下是合并五折折外预测的结果，不能理解为独立留出测试集成绩。

| 类别 | 人工复核数 | 精确率 | 召回率 |
|---|---:|---:|---:|
| 地震 | 40 | 0.348837 | 0.375000 |
| 爆破 | 35 | 0.655172 | 0.542857 |
| 交通 | 30 | 0.152174 | 0.233333 |
| 噪声 | 634 | 0.917874 | 0.899054 |

宏平均 F1 为 **0.5119432108**，同快照规则基线宏平均 F1 为 **0.0799006659**。混淆矩阵的行是真实类别、列是预测类别：

| 真实／预测 | 地震 | 爆破 | 交通 | 噪声 |
|---|---:|---:|---:|---:|
| 地震 | 15 | 4 | 0 | 21 |
| 爆破 | 7 | 19 | 0 | 9 |
| 交通 | 1 | 1 | 7 | 21 |
| 噪声 | 20 | 5 | 39 | 570 |

发布门槛仍为每类至少 20 个已复核事件、宏平均 F1 至少 0.70、每类召回率至少 0.50。本候选的宏平均 F1、地震召回率和交通召回率不达标，因此不能发布为活动模型。高噪声召回率不能替代地震和交通的识别证据。

已核验模型重新加载和单样本预测，概率和为 1。模型文件为 [model.bundle](D:/das-sentinel-runtime/app-data/artifacts/models/cce64999-eb01-4b47-8544-1301ebd6b33c/1/model.bundle)，SHA-256 为 `7534329a87df0f0de9ada30c785437da3e99043dad471c5381cf1d01e41c64b5`。训练快照哈希为 `3da3d0e9da317b185d291030cbf2ddf8d1ce44446c92c7f8f3089cf052ecd0c1`。证据文件：[加载核验](D:/das-sentinel-runtime/manual-work/first-model-20260914/api-candidate-verification.json)、[完整指标](D:/das-sentinel-runtime/manual-work/first-model-20260914/api-candidate-metrics.json)。这些指标不参与下面的候选筛选或数据划分。

## 冻结 DASNet 编码器的本地适配

采用作者发布的 [DASNet-v1](https://github.com/AI4EPS/models/releases/tag/DASNet-v1) 权重。官方模型是面向海底光缆声学与地震信号的 Mask R-CNN，不能直接把其鲸鸣、船舶或震相类别当作本地地震／爆破／交通／噪声标签。本实验加载其已预训练的 ResNet50 主干并冻结参数，只在本地复核样本上训练地震／爆破／交通／噪声四类线性分类头；这不是从零预训练，也不是完整检测器微调。

权重位于 [dasnet_v1.pth](D:/das-sentinel-runtime/pretrained-models/DASNet-v1/dasnet_v1.pth)，加载器先核对 SHA-256 `f1b0e1ee00a5a4a818b4bcfb70205bcbafd0db38acfa6d04537456576ae9c590`，再以 `weights_only=True` 和限定的 `argparse.Namespace` 白名单读取。主干参数严格匹配，不允许不完整加载后继续。

输入固定为 **60 秒、100 Hz、`832×6000`** 的有限 float32 波形，物理距离为 **0–3324 m、间距 4 m**，单位为 `(nm/m)/s * Hz/m`。站点光缆总长为 3403 m，但本批实际文件网格只到 3324 m；不外推、拉伸或声称已覆盖全部 3403 m。先建立原始、2–10 Hz 带通和大于 10 Hz 高通三视图，沿每个通道的完整时间轴归一化并裁剪至 [-3, 3]。无效和悬空通道通过识别掩码置零，地表有效通道保留。

标准 60 秒窗口不覆盖 0.01 Hz 长周期识别任务：0.01 Hz 的一个周期就是 100 秒。独立低频流程使用更长窗口和滤波保护区，不能继承本分类头的训练结论。低频 5 Hz 视图也无法表达上述 2–10 Hz 与大于 10 Hz 频带，不送入本编码器。

编码器输入为 **3×6000×256**：仅压缩距离轴，保留全部 6000 个时间点，避免通过时间缩放损失高于 10 Hz 的波形信息。随后使用已声明的 ImageNet 均值／标准差。仅在目标人工 ROI 与有效非悬空通道的交集内，聚合 layer2、layer3 的加权均值与最大值，形成 **3072 维 float32** 向量。池化保留精确的 ROI／特征单元重叠权重；卷积感受野仍会引入周围上下文，所以不能声称每个维度只依赖标注框内部。

分类头输入显式转成 float64，再拟合仅用训练折统计量的 `StandardScaler` 和类别平衡逻辑回归。浮点精度与 JSON 参数重载推理保持一致，保存时必须通过预测重载检查。这是已明确改变空间分辨率和输出头的迁移适配，不是官方原生检测器的等价复现。

## 比较方案与可用性修订

七个候选已在 API 基线训练前声明：[配方锁](D:/das-sentinel-runtime/manual-work/first-model-20260914/comparison-recipe-lock.json)。它们是 `rf25`、去掉 `time_of_day_hours` 的 `rf24`、`extra24_leaf1`、`extra24_leaf3`，以及 `dasnet_C0.01`、`dasnet_C0.1`、`dasnet_C1`。树模型均为 300 棵；DASNet 三个候选只改变分类头正则化强度，不改变预训练主干。选择依据是开发集四折合并预测的宏平均 F1；距离最高分 0.01 以内按预先指定的简洁、无时钟优先顺序选择。留出测试集只在选择锁定后评估一次。

完整 739 条记录包含三种物理网格；当前 832 通道、4 m 网格兼容队列有 687 条。比较必须使用相同的成功导出事件，不能把 739 条 RF 的分数与更小的波形队列分数直接比较。所有子集继承全量清单的重叠合并日期组，不重新拆开同日或跨日共享读取上下文的记录。

元数据上下文预检仅有 **586 条**通过：地震 25、爆破 32、交通 22、噪声 507。原有兼容队列划分经过这次可用性筛选后，测试集地震变成 **0**、爆破 2、交通 3、噪声 52，开发集交通只剩 **19**，因此不能继续原划分。修订只读取文件可用性、人工标签计数和既有组名，没有使用模型特征或预测指标；原配方锁和原划分文件保留作为审计证据。

新的[可用性临时划分](D:/das-sentinel-runtime/manual-work/first-model-20260914/split-available-provisional.json)为测试 **95 条：地震 4、爆破 2、交通 2、噪声 87**；开发 **491 条：地震 21、爆破 30、交通 20、噪声 420**。开发集四个验证折都含四类。选择限制测试组为 3–8 个、至少三个有噪声的日期、开发每类至少 20 条并保留每类至少四个有标签日期组，再用类别计数接近 20% 留出比例的确定性规则选择。完整原因见[修订记录](D:/das-sentinel-runtime/manual-work/first-model-20260914/split-available-provisional-reason.json)。

随后[目标 ROI 预检](D:/das-sentinel-runtime/manual-work/first-model-20260914/export-roi-preflight.json)发现，兼容队列有 51 条 ROI 完全处于悬空通道，其中 19 条已因上下文不足排除，新增 **32 条均为噪声标签**。它们的半开通道范围完全早于索引 75，即没有任何距离至少 300 m 的识别通道；既有识别掩码正确拒绝这些监督样本，没有为增加训练数量放宽掩码，也没有改写人工标签。

完整导出和编码现已完成，实际成功集合与两项预检的交集完全一致：**554 条，地震 25、爆破 32、交通 22、噪声 475，来自 24 个实际仍有样本的原日期组**。证据为[最终导出报告](D:/das-sentinel-runtime/manual-work/first-model-20260914/export-report.json)和[冻结编码报告](D:/das-sentinel-runtime/manual-work/first-model-20260914/embedding-report.json)。未改变临时方案的组归属；95 条是 ROI 排除前的历史测试数量。

正式[可用样本划分](D:/das-sentinel-runtime/manual-work/first-model-20260914/split-available.json)已在离线模型拟合前冻结，SHA-256 为 `7cb1845b6f444121aba8ada2b840fbdc681a876123f75c2726a2e158cb66fd69`：

| 最终划分 | 地震 | 爆破 | 交通 | 噪声 | 合计 |
|---|---:|---:|---:|---:|---:|
| 开发集 | 21 | 30 | 20 | 402 | 473 |
| 留出测试集 | 4 | 2 | 2 | 73 | 81 |

开发集四个验证折在地震／爆破／交通／噪声顺序下分别为 3/3/4/33、2/1/10/32、1/1/3/30、15/25/3/307，每个训练折与验证折均含四类。日期分组、事件 ID 和缓存哈希已核验；开发集交通恰好满足 20 条门槛。

## 已完成的共同队列比较

七个预声明候选均在同一 473 条开发样本、同一四折日期分组上完成评估。下表是合并折外预测的宏平均 F1，不能解释成留出测试指标。完整逐类结果和折结果见[选择记录](D:/das-sentinel-runtime/manual-work/first-model-20260914/experiment-v1/selection.json)。

| 配方 | 输入／分类器 | 开发集四折宏平均 F1 |
|---|---|---:|
| `rf25` | 25 特征随机森林，含时钟 | 0.373232 |
| `rf24` | 24 特征随机森林，不含时钟 | 0.350415 |
| `extra24_leaf1` | 24 特征 ExtraTrees，叶节点最少 1 条 | 0.332172 |
| **`extra24_leaf3`** | **24 特征 ExtraTrees，叶节点最少 3 条** | **0.381857** |
| `dasnet_C0.01` | 冻结 DASNet＋线性头，C=0.01 | 0.377607 |
| `dasnet_C0.1` | 冻结 DASNet＋线性头，C=0.1 | 0.374902 |
| `dasnet_C1` | 冻结 DASNet＋线性头，C=1 | 0.356391 |

按事先冻结的选择规则锁定 `extra24_leaf3`。最佳 DASNet 配方为 `dasnet_C0.01`，开发集 F1 与胜出模型接近，但没有显示预训练优势；本轮也没有对 DASNet 再运行留出测试来挑选更好结果。含时钟 RF25 的分数高于 RF24，只能说明这批分组数据上的现象，不能据此认定时钟携带可泛化的物理信号。

锁定后仅对胜出模型进行一次 **81 条留出测试**，宏平均 F1 为 **0.5883970489604292**。该评估模型仅使用 473 条开发样本拟合；[评估记录](D:/das-sentinel-runtime/manual-work/first-model-20260914/experiment-v1/evaluation.json)保留逐事件预测和[评估模型](D:/das-sentinel-runtime/manual-work/first-model-20260914/experiment-v1/evaluation-model.skops)哈希。

| 类别 | 留出数量 | 精确率 | 召回率 |
|---|---:|---:|---:|
| 地震 | 4 | 0.666667 | 0.500000 |
| 爆破 | 2 | 1.000000 | 0.500000 |
| 交通 | 2 | 0.125000 | 0.500000 |
| 噪声 | 73 | 0.942029 | 0.890411 |

| 真实／预测 | 地震 | 爆破 | 交通 | 噪声 |
|---|---:|---:|---:|---:|
| 地震 | 2 | 0 | 0 | 2 |
| 爆破 | 0 | 1 | 0 | 1 |
| 交通 | 0 | 0 | 1 | 1 |
| 噪声 | 1 | 0 | 7 | 65 |

测试 F1 仍低于 0.70，且交通误报明显、少数类样本过少，不能作为部署通过证据。API v1 的 0.511943 来自不同的 739 条记录和五折验证；这里的开发四折及 81 条测试使用更小且物理输入一致的队列，三个分数不能直接排序来宣称模型进步。

## 最终离线产物

评估后，按已锁定配方用全部 **554 条**共同样本重拟合并保存两个离线候选：

| 产物 | 作用 | 重载预测最大误差 |
|---|---|---:|
| [final-extra24_leaf3.skops](D:/das-sentinel-runtime/manual-work/first-model-20260914/experiment-v1/final-extra24_leaf3.skops) | 本轮选中的 24 特征 ExtraTrees | 1.11×10⁻¹⁶ |
| [final-dasnet_C0.01.json](D:/das-sentinel-runtime/manual-work/first-model-20260914/experiment-v1/final-dasnet_C0.01.json) | 最佳 DASNet 的标准化参数与四类线性头，须配合已核验主干 | 0 |

两者状态均为 `offline_candidate`，精确哈希、类别顺序和重载证据见[产物清单](D:/das-sentinel-runtime/manual-work/first-model-20260914/experiment-v1/final-artifacts.json)。最终重拟合已包含原测试样本，**没有额外独立泛化测试分数**；上述 0.588397 仅属于开发集拟合的评估版 ExtraTrees。保存 DASNet 头不意味着它赢得了比较或通过独立留出验证。两种架构均未冒充现有 RF bundle，未自动注册或发布。

最终[命令行验收](D:/das-sentinel-runtime/manual-work/first-model-20260914/experiment-v1/cli-verification.json)也已通过：对实际事件 `00aa255f-544a-4045-bb81-0df06eafaeec`，ExtraTrees 与保存后缓存推理的概率最大误差为 `2.78e-17`；DASNet 从真实 NPZ 重新编码后与缓存推理概率完全一致。两次 CLI 均正常退出。这验证产物可重新加载并用于推理，不增加泛化准确率证据。[验证图](D:/das-sentinel-runtime/manual-work/first-model-20260914/experiment-v1/validation.png)已完成目视检查。

## 复现入口与边界

实际脚本和审计位于 [first-model-20260914](D:/das-sentinel-runtime/manual-work/first-model-20260914)，依赖版本记录见 [runtime-requirements.txt](D:/das-sentinel-runtime/manual-work/first-model-20260914/runtime-requirements.txt)。以下是本轮实际入口记录，不要对已完成的目录重复执行。完成输出采用拒绝覆盖或哈希检查，重做实验应先建立新的独立运行目录并重新冻结输入。

```powershell
$repoRoot = 'D:\das-sentinel-runtime\waterfall-range-previews-git-20260901-2\.worktrees\jjk-scan-progress'
$runRoot = 'D:\das-sentinel-runtime\manual-work\first-model-20260914'
$readerPythonExe = 'D:\das-sentinel-runtime\.venv\Scripts\python.exe'
$pythonExe = 'D:\das-sentinel-runtime\.venv-first-model\Scripts\python.exe'
$env:PYTHONPATH = "$repoRoot\backend\src"
$env:PYTHONIOENCODING = 'utf-8'

# 波形导出使用含生产读取器依赖的环境；冻结编码器使用隔离科学计算环境。
& $readerPythonExe "$runRoot\export_cohort.py"
& $pythonExe "$runRoot\extract_embeddings.py"

# 仅在完整导出已验收、正式 split-available.json 已冻结后执行。
if (-not (Test-Path -LiteralPath "$runRoot\split-available.json")) {
    throw '正式可用样本划分尚未冻结，不能开始模型选择。'
}
& $pythonExe "$runRoot\experiment.py" select --embeddings "$runRoot\embeddings.npz"
& $pythonExe "$runRoot\experiment.py" evaluate
```

`select` 只计算开发折指标，保存选择结果；`evaluate` 检查划分、脚本和缓存哈希，拒绝重复启动同一留出评估。[predict_first_model.py](D:/das-sentinel-runtime/manual-work/first-model-20260914/predict_first_model.py)提供已保存离线产物的推理入口。以下使用本次实际导出的事件演示，预测概率未经校准；这是加载与推理使用方法，不是新增测试分数。

```powershell
# 沿用上文 repoRoot、runRoot、pythonExe 与 PYTHONPATH。
$eventId = '00aa255f-544a-4045-bb81-0df06eafaeec'
# 此事件复用了已核验的标准导出，实际路径以 samples-index.json 为准。
$sampleDir = "D:\das-sentinel-runtime\training-datasets\jjk-waveform-20260914\standard-validation\$eventId"
& $pythonExe "$runRoot\predict_first_model.py" --recipe dasnet_C0.01 `
    --sample $sampleDir --event-id $eventId

# 使用实际 CLI 验收保存的同一事件特征 JSON。
& $pythonExe "$runRoot\predict_first_model.py" --recipe extra24_leaf3 `
    --features-json "$runRoot\experiment-v1\smoke-features.json"
```

评估单位仍是人工事件 ROI，不是已经核实的独立地震震源。噪声占比高，2026-07-28 集中了大量样本和爆破标签；同源高包含度的不同标签 ROI 已列入审计，未自动改标或剔除。最终测试只有两个交通、两个爆破样本，一次错误就会使对应召回率改变 50 个百分点。本轮完成了可复现的候选训练与初步比较，尚未取得可靠部署或预训练模型优越性的证据。
