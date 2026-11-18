# 回归报告撰写工作表

本文件是组员自行撰写的工作清单和证据索引，不是可直接提交的报告正文。
报告要求由小组撰写；请理解代码、检查表图后使用自己的表述。

## 方法部分需要自己回答

- 为什么 Ridge 与 MLP 属于适合比较的不同模型家族？
- 公共 22 特征中，lag、天气、时间编码分别表示什么？哪些值在预测时已知？
- 为什么午夜减 30 分钟后再定义市场日？
- 输入与目标如何缩放，何时拟合，预测怎样恢复 MW？
- 2014 如何用于选型？为什么它不再是完全独立的最终测试？
- 为什么关闭随机内部 early stopping？训练集内 shuffle 与随机时间切分有什么区别？
- 主种子为何固定，三种子诊断能与不能支持什么结论？

## 数字和图表证据

| 证据 | 文件 | 自行解释的问题 |
| --- | --- | --- |
| 四方法 MAPE / MAE | outputs/stage1/demand_validation.csv | 最优方法相对两个 baseline 与另一算法改善多少？ |
| 超参数比较、训练与验证差距 | outputs/stage1/demand_tuning.csv | 网络更大、训练更久是否更好？是否存在过拟合迹象？ |
| 种子诊断 | outputs/stage1/demand_seed_stability.csv | 结论是否依赖某次初始化？ |
| 特征消融 | outputs/stage1/demand_feature_ablation.csv | 移除不同组特征时误差如何变化？两种算法是否一致？ |
| 总体成绩图 | outputs/stage1/demand_figures/validation_scores.png | 两个指标单位不同，分别用独立纵轴。 |
| 预先固定的夏冬周 | outputs/stage1/demand_figures/seasonal_week_predictions.png | 峰、谷和日变化是否存在系统性偏差？不要只靠目测下结论。 |
| 月度区间 | outputs/stage1/demand_figures/monthly_mape_intervals.png | 半小时独立区间与按日 bootstrap 宽度有何差别？ |
| 真实值和预测散点 | outputs/stage1/demand_figures/observed_vs_predicted.png | 高需求与低需求是否表现不同？ |
| 训练损失 | outputs/stage1/demand_figures/training_loss.png | 优化是否停止、达到上限还是满足容差？它不是验证 MAPE。 |
| 消融图 | outputs/stage1/demand_figures/feature_ablation.png | 特征作用依赖模型与固定训练设置，不能当因果作用。 |
| 分时段误差 | outputs/stage1/demand_figures/error_by_slot.png | 哪些半小时更难，是否与高峰和低谷一致？ |

## 区间解释必须保留的限制

- 此处仅为 2014 验证期区间，不能替代 D 的 2015–2026 历史退化分析。
- 月 MAPE 是当月观测误差均值；区间解释为类似当月条件下预期误差的不确定性，依赖月内过程相对稳定。
- IID 正态区间为均值 +/- 1.96 * 半小时误差样本标准差 / sqrt(n)。
- day bootstrap 每次有放回抽取与当月相同数量的整天，2000 次，取 2.5% 与 97.5% 分位数。
- 每天 48 条且无缺失，因此对每日 MAPE 重采样等价于拼接整天的 48 条误差后求均值。
- 按日 bootstrap 保留日内相关，但仍将天视为独立、可交换，不能充分处理连续热浪等跨日相关。
- 两个月区间是否重叠，不是对退化的正式显著性检验。
- 区间没有计入重复使用 2014 选型带来的选择不确定性，也不是单个需求值的预测区间。

## 留给小组其他部分

- D：四算法完整历史部署、检测器、定期和报警重训、累计预算与留出期复核。
- C：日类型及其与预测误差的关联。
- E：最终策略、训练窗口、双任务模型包、live-week 预估和全文整合。

## 可核对的算法参考

- MLPRegressor: https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html
- Ridge: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
- TransformedTargetRegressor: https://scikit-learn.org/stable/modules/generated/sklearn.compose.TransformedTargetRegressor.html
- StandardScaler: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html

组员应自行阅读并按课程要求整理引用，图表编号与正文最终统一。
