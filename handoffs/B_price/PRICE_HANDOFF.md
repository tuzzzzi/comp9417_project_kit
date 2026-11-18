# B 分类模型交接

## 运行

在仓库根目录使用 Python 3.11，安装 requirements.txt 中的依赖。

```bash
python -m experiments.validate_models --task price --data path/to/nem_weather_halfhourly_2010_2026.csv
python -m experiments.plot_price_results
```

本机已在本目录 .venv 中配置 Python 3.11.18，所有固定版本依赖均已核对。可用 .venv/bin/python 替换 python。本次数据路径为 ../comp9417_project_kit/data/nem_weather_halfhourly_2010_2026.csv。

## 文件

- experiments/price_models.py 提供 logistic 和 hist_gbc，每次 build_models() 返回新模型，可直接 fit/predict。
- experiments/plot_price_results.py 读取统一验证入口生成的 CSV，生成对比图和类别统计。
- outputs/stage1/price_validation.csv 保存四种方法的评分，单位为百分数。
- outputs/stage1/price_predictions.csv 保存 2014 年逐半小时预测，共 17,520 行。
- outputs/stage1/price_class_balance.csv 保存预测类别比例、两类召回率和混淆矩阵计数。
- outputs/stage1/price_figures/ 保存指标图和验证期第一周预测图。

## 设置与验证记录

共用 22 个输入特征，训练集 69,792 行（2010–2013 年，扣除首周滞后窗口）。
验证集覆盖 2014 年 365 个市场日，每日 48 行。
预测标签为价格严格高于前一市场日均价时取 1。
训练正类比例 45.92%，验证正类比例 48.62%。

HistGradientBoostingClassifier 使用 learning_rate=0.1、max_iter=100、
max_leaf_nodes=15、min_samples_leaf=20、class_weight="balanced"、
early_stopping=False、random_state=42。
这是单套固定参数实验。关闭 early stopping 以避免随机拆分时序训练数据。
Logistic Regression 使用组长原有的标准化 pipeline 和 balanced 权重。

| 方法 | Accuracy (%) | Balanced accuracy (%) |
| --- | --- | --- |
| yesterday | 73.90 | 73.99 |
| last_week | 74.25 | 74.28 |
| logistic | 70.72 | 70.41 |
| hist_gbc | 79.20 | 79.36 |

按组内主要指标推荐 hist_gbc 进入适应实验。历史回测需要同时评估两个冻结分类模型。
本次输出仅覆盖 2014 年验证，2015–2026 年由历史回测流程生成。
这些验证结果用于模型选择，后续年份表现需要另行评估。

已检查输入无缺失、预测时间戳唯一、每天 48 行，以及所有真实标签和预测均为 0/1。
图中展示的时段固定为验证集第一周，未根据预测效果挑选。
报告正文需结合实际结果由组员撰写。
