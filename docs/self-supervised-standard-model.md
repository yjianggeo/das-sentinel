# JJK 常规频段自监督模型

本轮按用户批准，已完成**非低频的 0.5–40 Hz 自监督训练及四类比较**。无标签数据分为 **106 个训练窗口、8 个验证窗口**；验证窗口全部来自与训练分离的日期 **2024-05-27**。训练后固定验证 MSE 为 **0.897161**，但预注册的全预算开发集四折分类 F1 仅 **0.288603**，低于随机编码器对照 **0.355043** 和 24 特征 ExtraTrees **0.381857**。因此本轮没有改善整体四分类，不应发布，也不能宣称节省标注成本。编码器和四类头已保存为离线候选；原人工标签、上一轮模型、API 候选与 worker 保持不变。

## 输入和识别范围

保存的数据为 **60 秒、100 Hz、832×6000 的 float32 全通道数组**，实际距离 **0–3324 m、间距 4 m**，单位为 `(nm/m)/s * Hz/m`。站点光缆总长为 3403 m，但本批实际文件网格不覆盖最后一段，不能拉伸坐标或声称已读取全部 3403 m。

先在完整 60 秒数组上进行 0.5–40 Hz 滤波，再逐通道去均值、RMS 归一化和裁剪至 ±6。保存数组保留全部通道；模型以既有识别掩码排除无效通道及 0–299 m 悬空段，保留地表和井下有效通道。无效区域既不作为可见输入，也不作为重建目标。缺失上下文、错误网格或时间轴异常应报告失败，不填零伪装成有效波形。

本模型不处理 600 秒／5 Hz 低频视图，也不代表 0.01 Hz 长周期识别能力；0.01 Hz 的单周期为 100 秒，不能由一个 60 秒标准窗替代。100 Hz 采样也不保留原始 50 Hz 以上信息。

## 独立无标签来源

来源为已有原始文件索引，不按地震、爆破、交通或噪声标签挑选预训练窗口。4031 个源文件索引中，排除全部 554 条监督样本所在的 24 个 UTC 日期及整日两侧各 611 秒，并额外核对最长读取上下文，剩 **2010 个候选文件、9 个候选日期**。这只是待验证源池，不能称其全部波形已经可用；原索引还记录两个不可读取的文件。

| UTC 日期 | 候选文件数 | 初步用途 |
|---|---:|---|
| 2024-04-17 | 60 | SSL 训练 |
| 2024-04-22 | 60 | SSL 训练 |
| 2024-05-27 | 516 | SSL 验证 |
| 2024-05-31 | 469 | SSL 训练 |
| 2024-06-10 | 865 | SSL 训练 |
| 2026-06-04 | 13 | SSL 训练 |
| 2026-06-14 | 3 | SSL 训练候选，无完整上下文时排除 |
| 2026-06-15 | 14 | SSL 训练 |
| 2026-08-31 | 10 | SSL 训练 |

每个日期抽查首、中、末共 27 个文件的元数据，全部匹配目标网格与单位；其中一个为 5000 Hz，其余抽样为 1000 Hz。本轮导出只接收已核验的 1000 Hz 源。每个训练日期最多抽取 24 个窗口，验证日期最多 8 个，按时间均衡、固定顺序选择；失败如实记录，不依据模型表现补选。

最终 **115 次尝试成功 114 个完整窗口**，唯一失败为 5000 Hz 源被既定契约拒绝。训练窗口 **106 个**，来自七个日期；SSL 验证 **8 个**，固定为 2024-05-27。实际逐日数量为：2024-04-17 24、2024-04-22 24、2024-05-27 8、2024-05-31 24、2024-06-10 24、2026-06-04 5、2026-06-15 1、2026-08-31 4。2026-06-14 没有足够完整读取上下文，未生成窗口。

已复验实际涉及的 **312 个来源指纹**并重新加载全部 **114 个**样本，确认数组均为 832×6000、标注列表为空及日期隔离。完整记录见[导出报告](D:/das-sentinel-runtime/manual-work/self-supervised-standard-20260914/export-report.json)和[最终样本索引](D:/das-sentinel-runtime/training-datasets/jjk-self-supervised-standard-20260914/samples-index.json)。这项实际验收仅覆盖本批窗口及其来源，不代表全部 2010 个候选文件都经过完整波形验证。

SSL 样本没有人工类别字段，标注列表为空且 `supervised_eligible=false`。源码指纹、源区间、实际读取区间、划分和失败原因均保留在[源池审计](D:/das-sentinel-runtime/manual-work/self-supervised-standard-20260914/source-pool-audit.json)及新数据集索引中。所有监督日期连同保护区都避开，因此同一自监督编码器可以用于下游开发集四折比较，而不接触其中任何验证日期波形。

## 本地轻量掩码重建模型

本轮是**自行实现的轻量卷积掩码自编码器**，共 **129681 个参数**。不加载 DAS-MAE 官方权重，也不能把本实现称作 DAS-MAE 论文模型的复现。它学习用可见波形恢复被遮挡的真实波形，不使用人工事件类别进行预训练。

每个输入块为 64 通道×512 个采样点，不缩放时间轴；以 8 通道×64 点为遮挡块，随机遮挡 50%。输入有两个平面：可见标准化信号和可见掩码。编码器采用三层 5×5、步长 2 的卷积，通道数为 16／32／64，结合 GroupNorm、GELU；解码器用三层转置卷积恢复尺寸，无跳接。均方误差只计算被遮挡且真实有效的像素，不能通过重建无效零值获得虚假低损失。

预先固定 CPU 最多 4 线程、batch 32、AdamW 学习率 0.001、权重衰减 0.0001、种子 20260914，最多 20 轮。每个窗口抽取 16 个固定种子的时空块；训练遮挡逐轮随机，验证块与遮挡固定。至少训练 5 轮，验证损失连续 5 轮无相对 0.1% 改进则早停；保存真正最低验证损失的权重。只按无标签验证重建损失选择编码器，不看分类或历史测试指标。

分类阶段冻结编码器，在原坐标系以 64×512 块覆盖所有与人工 ROI 相交的区域，尾部仅作无效 padding。按 ROI 和有效通道的精确面积聚合均值、最大值，得到 **128 维**特征。不能把整个 60 秒强制标成目标事件的单一类别；卷积感受野仍包含邻域上下文。缓存用 float32，标准化和四类线性头训练／JSON 重载计算统一使用 float64。

## 对照和标签预算

对照包括同架构的随机初始化编码器、训练后的 SSL 编码器，二者各比较 `C=0.01/0.1/1` 的类别平衡逻辑回归；另有不含时钟的 24 特征 ExtraTrees、叶节点最少 3 条基线，共七个预声明配方。分类标准化只拟合每折训练标签，随机初始化权重在预训练前保存，不能把资源基准中临时训练过的权重当作随机对照。

沿用 554 条已验证监督 ROI：开发集 473 条，历史对照 81 条。开发集按原日期组四折比较，训练标签预算为 25%、50%、100%，每折每类用相同固定顺序嵌套取样，至少保留每类一条，验证折标签保持完整。仅按 100% 开发折合并预测宏平均 F1 选择正则化强度，距最高分 0.01 内优先更强正则。

**25% 预算只是每折训练标签数量的离线模拟，不等于减少总人工成本 75%。** 完整验证标签和 100% 预算的参数选择仍消耗人工标注。应同时报告每折实际训练数量、完整验证数量和各预算结果，不能把小预算拟合当作生产每类 20 条门槛已经满足。

原 81 条已在上一轮查看并用于全量重拟合，因此只作配方锁定后的一次**历史对照**，不是新的独立测试，也不用于本轮选型。现有额外日期标签不能组成新的四类测试。最终 SSL 分类头已用全部 554 条标签拟合，不能为该最终产物宣称额外独立泛化成绩。重建误差下降与地震分类提高是不同结论，必须依据同预算、同分组的分类对照判断是否有收益。

## 分类比较结果

全部 554 条监督 ROI 已分别用随机初始化和 SSL 编码器提取配对特征。以下为同一 473 条开发样本、同一四折和嵌套训练子集上的合并折外宏平均 F1；每个预算都使用完整验证标签。完整逐事件预测和每折标签清单见[选择记录](D:/das-sentinel-runtime/manual-work/self-supervised-standard-20260914/comparison/selection.json)。

| 配方 | 25% 训练预算 | 50% 训练预算 | 100% 训练预算 |
|---|---:|---:|---:|
| 随机编码器 C=0.01 | 0.306971 | 0.330408 | 0.323667 |
| 随机编码器 C=0.1（选定对照） | 0.326521 | 0.337902 | 0.355043 |
| 随机编码器 C=1 | 0.352530 | 0.330300 | 0.364196 |
| SSL C=0.01（选定 SSL 头） | 0.339804 | 0.351997 | 0.288603 |
| SSL C=0.1 | 0.368145 | 0.331008 | 0.253210 |
| SSL C=1 | 0.352475 | 0.311136 | 0.234400 |
| 24 特征 ExtraTrees | 0.392377 | 0.374363 | 0.381857 |

每折实际训练标签数量如下，顺序为原四个开发折；不能将四折数量相加当作独立标注人数。

| 预算 | 每折训练条数 | 每折验证条数 |
|---|---|---|
| 25% | 109、109、111、33 | 43、45、35、350 |
| 50% | 216、215、220、63 | 43、45、35、350 |
| 100% | 430、428、438、123 | 43、45、35、350 |

选择按事先规定的 100% 预算结果进行。SSL 选中 C=0.01；随机编码器 C=1 的原始最高 F1 为 0.364196，但 C=0.1 距其不足 0.01，故按更强正则规则选中 C=0.1，而不是事后改动标准。部分小预算设置中 SSL 高于随机对照，但均未超过同预算 ExtraTrees，且完整预算结果更差，不能据此宣称自监督带来稳定收益或减少总人工成本。

| 100% 开发折合并召回率 | 地震 | 爆破 | 交通 | 噪声 |
|---|---:|---:|---:|---:|
| SSL C=0.01 | 0.190476 | 0.133333 | 0.500000 | 0.626866 |
| 随机编码器 C=0.1 | 0.238095 | 0.266667 | 0.250000 | 0.776119 |
| 24 特征 ExtraTrees | 0.000000 | 0.666667 | 0.350000 | 0.825871 |

这些结果也表明现有基线并未解决地震召回不足的问题。本轮不降低发布门槛、不替换原活动策略。

锁定配方后，对旧 81 条仅作一次历史对照：[历史结果](D:/das-sentinel-runtime/manual-work/self-supervised-standard-20260914/comparison/historical-results.json)。SSL、随机编码器、ExtraTrees 的历史 F1 分别为 **0.318398、0.401471、0.588397**；SSL 历史地震召回率为 **0**，爆破／交通分别为 0.5，噪声为 0.780822。历史队列只有 4 条地震、2 条爆破、2 条交通和 73 条噪声，又曾参与上一轮使用，不能解释成新独立测试或继续据此调参。这些数值只属于用 473 条开发标签拟合的历史评估模型。

## 已保存的离线产物

最终三个分类器均用全部 **554 条标签**重拟合，包含历史对照样本，没有新独立测试成绩。主要 SSL 产物为[best-encoder.pt](D:/das-sentinel-runtime/manual-work/self-supervised-standard-20260914/best-encoder.pt)与[final-ssl_C0.01.json](D:/das-sentinel-runtime/manual-work/self-supervised-standard-20260914/comparison/final-ssl_C0.01.json)，两者必须配套使用。分类头 SHA-256 为 `41e29c848e9789b6894a45e47d38be2e45e9ccadb9645523c32744bc443a38b0`。

同时保存[随机编码器分类头](D:/das-sentinel-runtime/manual-work/self-supervised-standard-20260914/comparison/final-random_C0.1.json)和[ExtraTrees 对照](D:/das-sentinel-runtime/manual-work/self-supervised-standard-20260914/comparison/final-extra24_leaf3.skops)。SSL 头、随机头、树模型保存重载的概率最大误差分别约 `5.55e-16`、`1.67e-15`、`1.67e-16`。完整哈希和状态见[产物清单](D:/das-sentinel-runtime/manual-work/self-supervised-standard-20260914/comparison/final-artifacts.json)，状态均为 `offline_candidate`；保存成功不等于识别性能验收通过。

## 验收证据

正式预训练在 CPU 上完成 **20 轮、277.156 秒**，最低固定验证重建误差出现在第 **20** 轮，达到预设最大轮数，没有因此扩展训练预算。结果见[训练报告](D:/das-sentinel-runtime/manual-work/self-supervised-standard-20260914/training-report.json)：

| 固定 SSL 验证集上的预测方式 | 遮挡有效像素 MSE |
|---|---:|
| 原始随机初始化自编码器 | 1.331591 |
| 恒为零的重建基线 | 0.992826 |
| 自监督训练最优权重 | 0.897161 |

最优权重已用 `weights_only=True` 严格重载，重载后验证 MSE 完全一致。[best-encoder.pt](D:/das-sentinel-runtime/manual-work/self-supervised-standard-20260914/best-encoder.pt)的 SHA-256 为 `b1463a0303339a9648efd99d1be3fca73a0a0a05a64204d521ccbb966d2205a0`；随机对照使用单独保存的[initial-encoder.pt](D:/das-sentinel-runtime/manual-work/self-supervised-standard-20260914/initial-encoder.pt)。这些误差衡量标准化波形的遮挡重建，不衡量地震、爆破、交通或噪声分类。

真实 CPU 基准使用 32 个标准波形块，预热后两个 AdamW 训练步均值 **0.240647 秒**，进程 RSS 约 **558 MB**；真实小 ROI 输出为有限的 128 维向量，编码耗时约 0.00838 秒。该基准仅证明小批计算可运行，不能外推成正式训练损失或识别效果。基准临时模型已丢弃，正式训练重新初始化。[完整基准记录](D:/das-sentinel-runtime/manual-work/self-supervised-standard-20260914/encoder-benchmark.json)保留输入与源文件校验信息。

组件 13 项、流程 9 项和导出器 7 项共 **29 项独立测试**已通过；主环境无需安装 PyTorch即可导入 NumPy／SciPy 预处理部分。正式完整导出、预训练、配对分类比较和保存重载已完成。最终 **5 次新进程真实 NPZ 推理**（4 次 SSL、1 次随机编码器）全部成功，与缓存推理的概率最大误差为 **0**；证据见[CLI 验收](D:/das-sentinel-runtime/manual-work/self-supervised-standard-20260914/comparison/cli-verification.json)。这验证重新编码和推理链路一致，不是额外准确率评估。分类头和树模型的保存重载误差均小于 `2e-15`。

[重建与分类验证图](D:/das-sentinel-runtime/manual-work/self-supervised-standard-20260914/comparison/validation.png)已生成并完成目视检查。[运行环境核验](D:/das-sentinel-runtime/manual-work/self-supervised-standard-20260914/final-runtime-verification.json)确认原有 89 个文件及 739 条最新人工标签未改，原 API 候选、worker 和健康状态保持。

[独立最终审查](D:/das-sentinel-runtime/manual-work/self-supervised-standard-20260914/independent-final-review.json)已重新计算全部 21 组开发验证和 3 组历史对照指标，逐项吻合；日期分组、嵌套预算、选参锁定时序、代码与数据哈希以及安全重载均通过，没有阻碍项。审查确认本轮分类负结果，未根据历史成绩重新选型。

运行资料位于 [self-supervised-standard-20260914](D:/das-sentinel-runtime/manual-work/self-supervised-standard-20260914)。方法和检查顺序见仓库中的设计、实施计划及进度记录；实际脚本为 [ssl_export.py](D:/das-sentinel-runtime/manual-work/self-supervised-standard-20260914/ssl_export.py)、[train_ssl.py](D:/das-sentinel-runtime/manual-work/self-supervised-standard-20260914/train_ssl.py)、[compare_ssl.py](D:/das-sentinel-runtime/manual-work/self-supervised-standard-20260914/compare_ssl.py)、[predict_ssl.py](D:/das-sentinel-runtime/manual-work/self-supervised-standard-20260914/predict_ssl.py)。波形读取使用原读取器环境，自监督训练使用现有隔离 CPU 科学计算环境；不修改现有服务依赖、原标签或已有模型。

## 使用已保存的离线模型

以下 PowerShell 命令使用实际 CLI 验收中的事件。该样本复用了上一轮标准导出目录，路径不能替换成新无标签窗口；四类推理仍要求唯一的人工事件 ROI。

```powershell
$repoRoot = 'D:\das-sentinel-runtime\waterfall-range-previews-git-20260901-2\.worktrees\jjk-scan-progress'
$runRoot = 'D:\das-sentinel-runtime\manual-work\self-supervised-standard-20260914'
$pythonExe = 'D:\das-sentinel-runtime\.venv-first-model\Scripts\python.exe'
$env:PYTHONPATH = "$repoRoot\backend\src"
$env:PYTHONIOENCODING = 'utf-8'
$eventId = '00aa255f-544a-4045-bb81-0df06eafaeec'
$sampleDir = "D:\das-sentinel-runtime\training-datasets\jjk-waveform-20260914\standard-validation\$eventId"
& $pythonExe "$runRoot\predict_ssl.py" --family ssl `
    --sample $sampleDir --event-id $eventId
```

入口自动核验 `best-encoder.pt`、`comparison/final-ssl_C0.01.json` 和训练报告／预处理代码哈希，安全加载权重，再从实际 NPZ 提取 128 维 ROI 特征。输出包括四类概率和 `offline_candidate` 状态；概率未经校准，不应作为自动告警阈值。需要随机初始化对照时，将 `--family ssl` 改为 `--family random`。本轮已完成可复现训练与负结果验证，保留产物供后续研究，不替换原服务模型。
