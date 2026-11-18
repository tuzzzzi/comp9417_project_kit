B 分类交接包

Price_Classification_Report.docx 为修改后的报告小节，含评分表和两张图。
合稿时统一图表编号、参考文献和整篇篇幅。

代码沿用组长仓库目录结构。
experiments/price_models.py 为修改文件，新增 hist_gbc。
experiments/plot_price_results.py 为新增绘图脚本。
其余 Python 文件为运行依赖。合并时先比较共用文件，避免覆盖队友的新修改。

使用 Python 3.11 独立环境，在本 README 所在目录运行：
python -m pip install -r requirements.txt
python -m experiments.validate_models --task price --data "课程CSV的完整路径"
python -m experiments.plot_price_results

结果保存到 outputs/stage1/，课程 CSV 由组员本地提供。
已在解压副本中用匹配依赖的 Python 3.11 重新运行，CSV 与原结果一致，绘图成功。
环境和数据不包含在本包中。

本包结果为 2014 年验证。hist_gbc 的 accuracy 为 79.20%，balanced accuracy 为 79.36%。
推荐 hist_gbc 进入后续适应实验。2015–2026 年预测由历史回测流程接入。
详细参数和输出说明见 PRICE_HANDOFF.md。
