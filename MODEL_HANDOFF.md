# Stage 1 模型协作指南

这是回归组和分类组共用的开发入口。先在组内确认下面的输入、时间切分和结果格式，再各自在独立文件中训练和比较模型。

## 固定的规则

| 共用项 | 规则 |
| --- | --- |
| 时间 | 用 2010–2013 年拟合模型和预处理，用 2014 年比较候选。首周积累滞后数据。午夜 00:00 归入前一个市场日。 |
| 输入 | 预测日的 12 列天气、时段和日历特征，以及更早市场日的需求和价格。两组先用同一份输入做对比。 |
| 需求目标 | 预测每个半小时的 NSW 用电量（MW）。 |
| 价格目标 | 预测每个半小时的实际电价是否严格高于前一市场日的全天均价。高于记 1，其余记 0。 |
| 基线 | 昨天和上周的同一半小时。价格基线都使用前一市场日的全天均价作阈值。 |
| 格式 | 模型提供 sklearn 风格的 `fit(X, y)` 和 `predict(X)`。每行输入对应一个预测值，需求返回 MW，价格返回 0/1。 |

输入、目标和两个基线由 [共用数据代码](experiments/stage1_data.py) 生成。[统一验证入口](experiments/validate_models.py) 负责拟合和评分。要研究新的共用特征，可以先对齐设置，再用同一份输入重跑两组的验证。

## 两组各自要做什么

| 分支建议 | 主要修改文件 | 组员交付 |
| --- | --- | --- |
| `demand-mlp` | [需求回归模型](experiments/demand_models.py) | 在 `build_models()` 中加入 `mlp`，保留 `ridge`。用课程数据训练 MLP，把特征缩放放进训练 pipeline，记录网络大小、随机种子及其他训练设置。交 Ridge、MLP 和两个基线的 MAPE、MAE，推荐 MAPE 最低的候选，并解释结果。 |
| `price-classification` | [电价分类模型](experiments/price_models.py) | 在 `build_models()` 中加入 `hist_gbc`，保留 `logistic`。比较两种方法的类别处理方式，使用 2014 年做选择。交两种模型和两个基线的 accuracy、balanced accuracy，推荐 balanced accuracy 最高的候选，并说明容易判错的情况。 |

`build_models()` 每次返回一份新字典，键是模型名，值是尚未拟合的估计器。Ridge 和 Logistic Regression 已作为可运行参考。组员在自己的分支提交模型代码与设置，在 Pull Request 里附上运行命令和评分表。

## 怎么跑

按[课程 README](README.md#1-create-a-new-environment-required) 配好 Python 3.11 和对应依赖。每位组员把课程发布的 CSV 放到项目的 `data/` 目录。

从项目根目录运行：

```powershell
python -m experiments.validate_models --task demand
python -m experiments.validate_models --task price
python -m experiments.validate_models
```

前两条分别跑一个任务，最后一条跑两个任务。结果保存在 `outputs/stage1/`：

- `demand_validation.csv`、`price_validation.csv` 是每个模型和两个基线的评分表。
- `demand_predictions.csv`、`price_predictions.csv` 是逐半小时结果。每行带有 `timestamp`、`market_day`、`actual` 和各方法的预测列。下游可用 `market_day` 对齐日型和误差。

每组的验收标准是同一个命令可以复现**两个候选加两个基线**的 2014 年结果。两个任务的候选都确定后，将四套模型配置和各任务的入选模型交给历史回测负责人。
