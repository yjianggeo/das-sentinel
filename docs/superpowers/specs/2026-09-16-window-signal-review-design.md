# 30 秒训练窗口的规则检测与双瀑布图复核

## 目标与已确认交互

用户明确保留 30 秒全通道窗口为唯一训练样本，使用传统信号检测筛选疑似信号，并确认“整窗标框＋点击查看局部预览”。地震、爆破、车辆仍按整窗进行三态多标签人工复核。局部框是证据，不创建新事件、训练样本或自动标签。

读取已为抗混叠重采样后的 100 Hz、全原生通道×3000点。保持数值、有效/识别掩码、输入签名、历史标签和模型契约。未带通瀑布图使用此数值，另一图使用已有带上下文的 0.5–40 Hz 零相位滤波结果。停止生成和显示波形折线图。这里“未滤波”不意味着省略降采样所需的抗混叠滤波。当前模型从保存的未带通数组计算整窗 RMS/峰值特征，图件不作为模型输入。

## 流程

1. 继续复用 screen_block 的多尺度 STA/LTA、分频能量、空间相干、轨迹及融合门槛；仅 qualified 证据进入局部框，弱证据保留审计，不额外调阈值或调用 AI。
2. 对同窗、时间及通道重叠的 qualified 证据合并为稳定编号局部区域，裁剪到 0–30 秒和原生通道边界。避免把弱触发、窗外数据和排除道变成新候选。合并只改变复核呈现，不改变既有 candidate/score 或输入签名。
3. 事件页默认查询已成功且 candidate=true 的疑似窗口，按已有目录优先级和证据分排序；保留推荐队列及全部窗口，背景窗口仍可人工标为负样本，未触发不能自动当成噪声。
4. 两张全通道瀑布图包含编号框，坐标以本地参考时间为 0 秒。点击框或区域列表，请求该分析版本的局部原始/滤波瀑布图，至少 3 秒并裁剪在本窗内。只用缓存数值生成局部图，不重复读取 NAS 或分析、不创建后台任务。

## 数据与 API 契约

- 不改 whole-window-v2 数值/检测算法或 compute_input_sha256。新增独立 visualization_version=2。
- 新图件 manifest：waveform/waveform.npz、evidence/evidence.json、metadata/metadata_visual_v2.json、waterfall/waterfall_raw_v2.png、waterfall_filtered/waterfall_filtered_v2.png、filtered_waveform/filtered_waveform_v2.npz。后者仅保存 float32 的 waveform[通道,3000]，用于快速局部滤波图；不混入训练输入。
- metadata/result 新字段 detection_regions: [{id(从1开始),start_sample,end_sample,peak_sample,channel_start,channel_stop,distance_start_m,distance_stop_m,score,detectors,evidence_count}]。距离为原生通道单元边界。
- plot_geometry: {axes:[left,top,width,height],distance_min_m,distance_max_m}，axes 为整张 PNG 归一化坐标（从左上起）。绘图与前端框共用此几何，所有通道保留，取消每5点直接抽取的图像时间抽样。
- GET event-windows 增加 candidate_only=false；筛选在数据库分页前执行，必须 succeeded 且 candidate=true。detail 增加 candidate、visualization_version、detection_regions、plot_geometry，旧结果返回空区域/null几何。
- GET event-windows/{window_id}/region-preview?analysis_id=...&region_id=1&filtered=false：认证、分析归属和状态、图件 SHA 校验、区域ID检查；返回 image/png，选中区域加适量上下文，最少3秒但不跨30秒主窗。无任意文件/距离输入。原始及滤波局部图随图类型切换，标注仍写整窗。
- API result 验证同时接受旧5项及新版6项图件，按 visualization_version 严格检查预期集合；验证滤波数值的 shape、finite、float32及元数据区域一致性。新 worker 只生成新版6项。

## 已有数据与上线

先在真实已成功窗口验证衍生图件：校验旧 manifest/SHA，按已保存来源重读连续上下文，只滤波及绘图、不重新检测；核对重读主窗与原 waveform.npz 数值一致。添加新版文件，不删除/覆盖旧图与旧元数据。数据库仅条件更新成功分析的 result_json，input_sha256、分析状态、训练数组及所有标签不变；保留备份及逐窗报告。

完成全部已有成功窗口的补图；新 worker 继续现有队列，不恢复取消任务。API/worker 如需切换，先备份并核对具体进程，保持单 worker；不推送或合并 GitHub，不重置既有 dirty 工作树。

补图允许从相邻成功半分钟的缓存数组组成 90 秒滤波上下文：必须同站点、算法、100 Hz、单位、网格和形状，逐份验证输入及 waveform SHA，邻窗主窗连同前后 10 秒抗混叠范围内的完整来源集合及元数据必须与当前窗口一致，否则回源。当前主窗始终逐点一致；相邻缓存因重采样分块导致的浮点舍入差异按数值容差处理，真实对照上下文最大误差 5.96e-8、滤波主窗误差 4.70e-11（原值峰值1.63），不能声称上下文逐点相同。此变化只用于展示衍生图件，训练原数组、原检测特征及标签签名不改。

## 验收

合成信号验证 qualified 合并/弱证据剔除/边界/排除道；100 Hz 原始数组及输入签名不变；两张瀑布图确有滤波差异且无 waveform_plot；API 分页筛选/旧manifest兼容/区域预览归属及SHA防护；UI 点击框、切图、异步切换、全屏键盘操作、多标签保存回归。真实源窗口验证双图和局部图并完成浏览器验收，不宣称检测结果已排除所有噪声或已完成模型训练。
