"""Generate technical handoff records and a worksheet, not report prose."""

import json
from pathlib import Path

import pandas as pd

from .stage1_data import ROOT


def table(frame, columns):
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in frame[columns].itertuples(index=False, name=None):
        lines.append("| " + " | ".join(f"{x:.6f}" if isinstance(x, float) else str(x) for x in row) + " |")
    return "\n".join(lines)


def main():
    output = ROOT / "outputs/stage1"
    config = json.loads((output / "demand_model_config.json").read_text())
    manifest = json.loads((output / "demand_run_manifest.json").read_text())
    checks = json.loads((output / "demand_checks.json").read_text())
    scores = pd.read_csv(output / "demand_validation.csv")
    tuning = pd.read_csv(output / "demand_tuning.csv")
    stability = pd.read_csv(output / "demand_seed_stability.csv")
    ablation = pd.read_csv(output / "demand_feature_ablation.csv")
    mlp_row = tuning.set_index("candidate").loc[config["selected_candidates"]["mlp"]]
    score_table = table(scores, ["model", "MAPE", "MAE", "MAPE_improvement_vs_yesterday_pct"])
    stability_table = table(stability, ["seed", "MAPE", "MAE", "epochs", "convergence_warning"])
    ablation_table = table(ablation, ["model", "removed_group", "n_features", "MAPE", "MAE"])
    readme = f"""# A 需求回归交接包使用说明

本包供 A、D、E 组内复现与整合使用。包含回归模型工厂、2014 验证结果、冻结模型和可导出图表。
不包含课程原始 CSV、Python 环境或可直接提交的双任务 submission.zip。

## 本机运行

```bash
cd {ROOT}
conda activate {ROOT.parent / '.venv-a'}
```

重新打开终端后先激活环境。也可直接使用 `{ROOT.parent / '.venv-a/bin/python'}`。
环境为 Python 3.11；课程 requirements.txt 中固定版本已由检查脚本核对。

## 从零复现本包

其他机器先创建 Python 3.11 独立环境，安装依赖：

```bash
python -m pip install -r requirements.txt
python -m experiments.reproduce_a --data /完整路径/nem_weather_halfhourly_2010_2026.csv
```

此命令会运行参数比较、固定种子诊断、特征消融、独立验证入口重训、绘图、校验和打包。
仅用发布 CSV，不调用外部数据或网络。依赖安装本身需要联网；训练代码不联网。
输出会覆盖本包已有的同名实验结果；需要保留旧实验时请先复制整个交接目录。

## 分步运行

```bash
python -m experiments.run_demand_study --data /完整路径/课程数据.csv
python -m experiments.validate_models --task demand --data /完整路径/课程数据.csv --output outputs/reproduction
python -m experiments.plot_demand_results
python -m experiments.check_demand_handoff --data /完整路径/课程数据.csv --reproduced outputs/reproduction
python -m experiments.write_demand_handoff
python -m experiments.package_demand_handoff
```

只重新出图可运行 `python -m experiments.plot_demand_results`，不会训练模型。
只按选定配置重建两个冻结模型：

```bash
python -m experiments.train_frozen_demand --data /完整路径/课程数据.csv
```

同一数据、版本和线程设置下重训预测应在数值容差内一致；训练耗时会变化。
完整依赖记录见 environment.freeze.txt，运行和数据哈希见 outputs/stage1/demand_run_manifest.json。

## 最先查看

1. DEMAND_HANDOFF.md：接口、参数、实际成绩和整合边界。
2. outputs/stage1/demand_validation.csv：四种方法的成绩。
3. outputs/stage1/demand_figures/validation_scores.png：成绩图。
4. outputs/stage1/demand_figures/seasonal_week_predictions.png：两段预先指定周的预测曲线。
5. REPORT_WORKSHEET.md：供组员自行撰写报告的证据位置和检查问题。

本包的成绩为模型选择所使用的 2014 验证成绩，不能视为未参与选择的未来测试成绩。
2015–2026 的历史部署、漂移检测、重训策略由 D 接续；最终双任务模型包由 E 整合。
"""
    destination = ROOT / "handoffs" / "A_demand"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "README.md").write_text(readme)
    handoff = f"""# A 需求回归技术交接记录

## 实验范围与固定规则

- 训练市场日：2010-01-08 至 2013-12-31，{manifest['train_rows']:,} 行；发布数据最初七天用于积累 lag。
- 验证市场日：2014-01-01 至 2014-12-31，{manifest['validation_rows']:,} 行，每天 48 行。
- 验证时间戳：2014-01-01 00:30 至 2015-01-01 00:00；午夜属于前一市场日。
- 固定 AEST UTC+10，不作夏令时转换。
- 两模型共用组内原始 22 特征；数据构造和官方指标代码未修改。
- 目标为 NSW 需求 MW；MAPE 单位 %，MAE 单位 MW，均越低越好。
- 2014 用于配置选择；保留 2010–2013 拟合的权重，不用 2014 重拟合，不访问 2015+ 标签进行选型。
- 只用课程 CSV；完整源文件 SHA256：`{manifest['data_sha256']}`。

## 模型与实际配置

Ridge：StandardScaler + Ridge(alpha={config['ridge_alpha']})，scaler 仅在训练样本拟合。
MLP：输入 StandardScaler + MLPRegressor，外层 TransformedTargetRegressor 对需求标准化并自动恢复 MW。

```json
{json.dumps(config['mlp_params'], indent=2)}
```

`shuffle=True` 仅打乱已经划定的 2010–2013 训练集内部的小批次顺序；不随机拆分训练、验证或测试日期。
`early_stopping=False` 不创建随机内部验证集；仍会依据训练损失容差结束优化。
主实验固定 seed=42；另外的 seed=0、2026 仅用于敏感性诊断，不选择最幸运种子。

参数比较：Ridge alpha=1、10、100；MLP 比较 (32,)、(64,)、(64,32) 及较强正则化，并增加 (64,) 的 1200 次迭代上限对照。
第一次 300 次上限试跑记录保存在 outputs/pilot_300；完整复现入口从头运行包含长训练对照的最终实验。
每个家族按 2014 MAPE 最小选择；同分依 MAE、候选名称稳定排序。
选定 MLP 候选：`{config['selected_candidates']['mlp']}`；实际训练 {int(mlp_row['epochs'])} epochs；收敛警告：{bool(mlp_row['convergence_warning'])}。
警告和每轮训练损失全部保留，不能把达到迭代上限描述为已经收敛。

## 2014 验证成绩

{score_table}

组内选型规则推荐进入适应实验的回归模型：`{config['recommended_model']}`。
两种算法的冻结版本都交给 D 做后续历史评估，不能只保留入选算法。
改善列是相对 yesterday MAPE 的百分比改善，不是百分点差。

## 随机种子敏感性

{stability_table}

仅三个预设种子，表示对初始化和训练批次随机性的敏感性；不是泛化误差置信区间。
MLP MAPE 的三种子均值为 {stability.MAPE.mean():.6f}%，样本标准差为 {stability.MAPE.std(ddof=1):.6f} 个百分点。

## 特征消融诊断

{ablation_table}

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

当前检查状态：{checks['status']}；完整记录见 outputs/stage1/demand_checks.json。
已独立复算分数、核对原始目标和 baseline，核对全部保存模型预测、训练 scaler 统计量与版本。
已抽查年初、冬季、年末三个日期：实际 history/weather 输入与离线特征一致；篡改未来市场值不改变预测特征。
未声称此 A 包已通过最终双任务 Linux 沙箱测试；E 仍需组装 Forecaster、train.py 并检查最终 submission.zip。

## 组内合并

- 核心新增/修改：demand_models.py、demand_config.json、run_demand_study.py、plot_demand_results.py、check_demand_handoff.py，以及复现和打包脚本。
- stage1_data.py、validate_models.py、price_models.py、harness/dayahead.py、requirements.txt 保留 B 交接版本；合并时不要覆盖队友更新后的公共文件。
- 本包附带 price_models.py 仅满足公共入口导入，不把 B 的结果归入 A。
- 初始训练和本次超参数实验属于 Stage 1，不计入 2015–2026 的六次部署重训预算。
"""
    (destination / "DEMAND_HANDOFF.md").write_text(handoff)
    worksheet = """# 回归报告撰写工作表

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
"""
    (destination / "REPORT_WORKSHEET.md").write_text(worksheet)
    print("Wrote A documents under handoffs/A_demand/")


if __name__ == "__main__":
    main()
