# 训练准备度与推荐复核

用户已确认“训练准备度检查＋地震优先复核队列＋相似事件分组”。延续现有分支和部署方式。

## 范围与约束

不训练或发布模型、不更改标签、不删除/合并数据库事件、不重跑扫描。原四类标注、普通事件队列和低频入口继续可用。
新增能力使用数据库已有事件/人工标签及本地已计算的目录到时，读取不触发NAS扫描。没有模型时不声称提供模型不确定性。
推荐只说明查阅顺序，相似组不等同独立地震，不自动继承标签。训练门槛保持不变，当前日期分组与真正独立事件的区别要明确。

## 训练准备度

GET /api/v1/sites/{site_id}/training-readiness，登录保护，只读。
返回site_id、reviewed_events、unreviewed_events、unsupported_label_count、classes（label/reviewed_count/utc_day_count/missing_count/minimum_days_missing）、
ready、blockers、warnings、folds（fold/train_counts/test_counts/missing_train_labels/missing_test_labels）、pipeline_counts、profile_counts、model_count、active_model_version。
ready仅表示当前训练结构门槛满足，不承诺训练/发布指标。检查必须匹配生产同一最新review排序、有限批准特征schema、每类20条、UTC日期分组、固定5折及四类覆盖；不调用fit。
在模型页显示每类数量、UTC日期、具体缺口、混合版本和低频独立标签说明；未就绪/加载失败时不误报可训练。刷新可以看到后续复核变化。

## 推荐队列与临时分组

GET /api/v1/review-batches?site_id=<id>&offset=0&limit=20，limit最多50，offset为推荐组偏移。
返回site_id、total_unreviewed、total_groups、offset、limit、reference_counts、catalog_window_count、warnings、groups。
每组包含id、representative（既有EventSummary）、members（至多20条EventSummary）、start_utc、end_utc、priority_score、reasons、selection_kind（earthquake/diverse/audit）、similar_earthquake_event_id。
只选择未复核事件；人工标签以最新复核为准，机器噪声也保留抽查机会。参考只用同站点、同pipeline/profile兼容的人工确认样本。
用有限、可解释的信号特征比较与已确认地震及噪声的距离；推荐分不是概率。无参考/坏特征时明示并保留候选。不得用时间、目录名、机器标签当训练标签。
同站点、pipeline/profile、源目录、空间重叠的近时（组起点起120秒内）且特征相近片段可归组；避免链式无限合并，最多20片段，允许展开逐条复核。跨时间的相似噪声通过推荐批次的多样性降低重复，不隐式合并为一次物理事件。
在完整站点池中确定分组/顺序，再分页；固定输入顺序和分页必须稳定、无重复。每20组优先10组地震线索、6组多样类型、4组可重复随机抽查；不足按剩余候选补齐。每个未复核事件最终都能在后续页找到。

## 目录线索

可选app-data/review-catalog-windows.json记录schema_version=1、windows数组；每项site_id/catalog_id/phase/start_utc/end_utc/source_url。
本地导入此前已计算的JJK目录到时，保留来源和时间窗。严格按site_id与UTC时间相交，仅作为推荐线索，无覆盖/文件缺失时不阻断普通相似排序；不猜测数据日期，不按目录名eq直接判地震。

## 前端与验收

原事件复核页增加“全部事件／推荐复核”切换，默认仍保留全部事件；推荐模式选站点、每批20组、原因和展开组内成员，复用原图件与人工标签表单。
当前批次不因保存一条标签而跳走；提供明确刷新/下一批，已复核项即时标识，下一次请求不再返回。切站点/模式、慢响应和卸载均隔离。
保存本批标签后，“下一批”改为“更新待复核批次”并请求offset=0；禁用旧上一批偏移，避免待复核池缩小造成漏看。未保存时正常翻页，显式刷新始终从0开始。
模型页加入准备度表和错误/刷新处理。原页面布局、排序与标签保存回归，API权限与只读性、最新标签、跨站点隔离、稳定分页/分组、版本隔离及真实只读评估均验证。
相关测试和构建通过后备份数据库，仅更新本地API/前端，保持worker不变。无Git提交/推送。
