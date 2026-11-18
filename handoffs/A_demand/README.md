# A 需求回归交接包使用说明

本包供 A、D、E 组内复现与整合使用。包含回归模型工厂、2014 验证结果、冻结模型和可导出图表。
不包含课程原始 CSV、Python 环境或可直接提交的双任务 submission.zip。

## 本机运行

```bash
cd /Users/q-chen/Documents/UNSW/2026T3/COMP9417/Project/A_demand_handoff
conda activate /Users/q-chen/Documents/UNSW/2026T3/COMP9417/Project/.venv-a
```

重新打开终端后先激活环境。也可直接使用 `/Users/q-chen/Documents/UNSW/2026T3/COMP9417/Project/.venv-a/bin/python`。
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
