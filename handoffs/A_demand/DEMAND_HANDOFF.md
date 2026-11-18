# A 需求回归技术交接记录

## 实验范围与固定规则

- 训练市场日：2010-01-08 至 2013-12-31，69,792 行；发布数据最初七天用于积累 lag。
- 验证市场日：2014-01-01 至 2014-12-31，17,520 行，每天 48 行。
- 验证时间戳：2014-01-01 00:30 至 2015-01-01 00:00；午夜属于前一市场日。
- 固定 AEST UTC+10，不作夏令时转换。
- 两模型共用组内原始 22 特征；数据构造和官方指标代码未修改。
- 目标为 NSW 需求 MW；MAPE 单位 %，MAE 单位 MW，均越低越好。
- 2014 用于配置选择；保留 2010–2013 拟合的权重，不用 2014 重拟合，不访问 2015+ 标签进行选型。
- 只用课程 CSV；完整源文件 SHA256：`c13711002906140eaf7cc67c1ce17a49facd359cf3d1a84722aaa3ccd647752c`。

## 模型与实际配置

Ridge：StandardScaler + Ridge(alpha=10.0)，scaler 仅在训练样本拟合。
MLP：输入 StandardScaler + MLPRegressor，外层 TransformedTargetRegressor 对需求标准化并自动恢复 MW。

```json
{
  "activation": "relu",
  "alpha": 0.0001,
  "batch_size": 256,
  "early_stopping": false,
  "hidden_layer_sizes": [
    64
  ],
  "learning_rate_init": 0.001,
  "max_iter": 300,
  "n_iter_no_change": 30,
  "random_state": 42,
  "shuffle": true,
  "solver": "adam",
  "tol": 1e-05
}
```

`shuffle=True` 仅打乱已经划定的 2010–2013 训练集内部的小批次顺序；不随机拆分训练、验证或测试日期。
`early_stopping=False` 不创建随机内部验证集；仍会依据训练损失容差结束优化。
主实验固定 seed=42；另外的 seed=0、2026 仅用于敏感性诊断，不选择最幸运种子。

参数比较：Ridge alpha=1、10、100；MLP 比较 (32,)、(64,)、(64,32) 及较强正则化，并增加 (64,) 的 1200 次迭代上限对照。
第一次 300 次上限试跑记录保存在 outputs/pilot_300；完整复现入口从头运行包含长训练对照的最终实验。
每个家族按 2014 MAPE 最小选择；同分依 MAE、候选名称稳定排序。
选定 MLP 候选：`mlp_64`；实际训练 300 epochs；收敛警告：True。
警告和每轮训练损失全部保留，不能把达到迭代上限描述为已经收敛。

## 2014 验证成绩

| model | MAPE | MAE | MAPE_improvement_vs_yesterday_pct |
| --- | --- | --- | --- |
| yesterday | 4.934955 | 394.422374 | 0.000000 |
| last_week | 4.933906 | 401.256094 | 0.021263 |
| ridge | 3.903412 | 311.306613 | 20.902797 |
| mlp | 3.409606 | 269.209621 | 30.909084 |

组内选型规则推荐进入适应实验的回归模型：`mlp`。
两种算法的冻结版本都交给 D 做后续历史评估，不能只保留入选算法。
改善列是相对 yesterday MAPE 的百分比改善，不是百分点差。

## 随机种子敏感性

| seed | MAPE | MAE | epochs | convergence_warning |
| --- | --- | --- | --- | --- |
| 42 | 3.409606 | 269.209621 | 300.000000 | True |
| 0 | 3.323419 | 261.529885 | 300.000000 | True |
| 2026 | 3.498284 | 274.472824 | 300.000000 | True |

仅三个预设种子，表示对初始化和训练批次随机性的敏感性；不是泛化误差置信区间。
MLP MAPE 的三种子均值为 3.410436%，样本标准差为 0.087436 个百分点。

## 特征消融诊断

| model | removed_group | n_features | MAPE | MAE |
| --- | --- | --- | --- | --- |
| ridge | none | 22 | 3.903412 | 311.306613 |
| mlp | none | 22 | 3.409606 | 269.209621 |
| ridge | weather | 10 | 3.885166 | 311.762414 |
| mlp | weather | 10 | 3.890713 | 308.894063 |
| ridge | calendar | 17 | 4.043633 | 323.138921 |
| mlp | calendar | 17 | 3.864805 | 304.851674 |
| ridge | demand_lags | 20 | 10.152092 | 769.923021 |
| mlp | demand_lags | 20 | 5.731932 | 443.660159 |
| ridge | price_history | 19 | 3.910476 | 311.774351 |
| mlp | price_history | 19 | 3.497455 | 277.221323 |

每项移除一组特征，保持该算法选定超参数和 seed 不变，并重新拟合训练期 scaler 和模型。
消融不用于更换最终公共 22 特征；它是固定优化预算和配置下的预测敏感性证据，不是因果贡献分解。
Ridge 与 MLP 对某组特征的收益可能不同，不能仅凭相关性宣称因果关系。

## 交给 D 的接口

复制 experiments/demand_models.py 和 experiments/demand_config.json，保留同一份 stage1_data.py。
缺失或不完整的配置会明确报错，不会静默换回默认网络。合并时这两个文件必须一起复制。

```python
from experiments.demand_models import build_models
from experiments.stage1_data import build_features

models = build_models()  # 每次都是新、未拟合的模型
model = models['mlp']   # 或 ridge
model.fit(past_X, past_y)  # 由 D 的合法训练窗口提供
next_X = build_features(history, weather_next)
predicted_mw = model.predict(next_X)  # 一维，每行一个 MW 值
```

冻结模型分别在 artifacts/demand_ridge_frozen.joblib、artifacts/demand_mlp_frozen.joblib。
它们仅包含 sklearn 标准估计器与预处理，可用 joblib.load 加载；勿只加载内部网络而丢弃缩放器。
`python -m experiments.train_frozen_demand --data /完整路径/课程数据.csv` 可按既定配置重建这两个文件，不再搜索超参数。

给 C 的长表为 outputs/stage1/demand_predictions_long.csv，列含 timestamp、market_day、model、y_demand、pred_demand。
model 包含 yesterday、last_week、ridge、mlp，每种 17,520 行。
日类型可事后合并用于误差分析，目标日真实日类型不能作为预测输入。

## 校验与边界

当前检查状态：passed；完整记录见 outputs/stage1/demand_checks.json。
已独立复算分数、核对原始目标和 baseline，核对全部保存模型预测、训练 scaler 统计量与版本。
已抽查年初、冬季、年末三个日期：实际 history/weather 输入与离线特征一致；篡改未来市场值不改变预测特征。
未声称此 A 包已通过最终双任务 Linux 沙箱测试；E 仍需组装 Forecaster、train.py 并检查最终 submission.zip。

## 组内合并

- 核心新增/修改：demand_models.py、demand_config.json、run_demand_study.py、plot_demand_results.py、check_demand_handoff.py，以及复现和打包脚本。
- stage1_data.py、validate_models.py、price_models.py、harness/dayahead.py、requirements.txt 保留 B 交接版本；合并时不要覆盖队友更新后的公共文件。
- 本包附带 price_models.py 仅满足公共入口导入，不把 B 的结果归入 A。
- 初始训练和本次超参数实验属于 Stage 1，不计入 2015–2026 的六次部署重训预算。
