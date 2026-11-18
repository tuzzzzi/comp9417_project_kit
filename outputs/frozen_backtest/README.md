# Frozen historical backtest: reading and reproduction

From the kit root, in the project Python 3.11 environment:

```bash
python -m experiments.backtest_frozen
```

The default run covers 2015-01-01 to 2026-10-31, 4,322 complete market days.
Ridge/MLP load A's 2010–2013 fitted artifacts. The two classifiers are fitted once
on the same 2010–2013 rows using B's fixed settings. No model is updated in Stage 2.
2014 is used only to check that the original validation scores are reproduced.

Every prediction is made from the previous seven completed market days and the
provided target-day weather. The seven-day truncation matches the full-history
shared features on the recorded boundary checks. Target-day market values are
revealed only after feature construction. Models receive no target-day day type.

## Files

| File | Meaning |
| --- | --- |
| `summary.csv` | Pooled scores for the whole simulation, 2015–2024, and 2025–2026.10. These are descriptive comparisons, not a new selection of the Stage 1 models. |
| `yearly_metrics.csv` | Official task-specific metrics per market year. 2026 has only January–October. |
| `monthly_metrics.csv` | Monthly metrics and both confidence intervals; intervals use demand MAPE or price accuracy, not balanced accuracy. |
| `daily_metrics.csv` | Per-day metrics for all methods; classification balanced accuracy follows the official single-class convention. |
| `*_predictions.csv.gz` | Full half-hour predictions, labels, timestamps and market days. `pd.read_csv` reads gzip automatically. |
| `*_monthly_intervals.png` | Each frozen algorithm vs both baselines, with IID and day-block 95% intervals. |
| `*_yearly.png` | Annual task metrics. |
| `demand_errors_by_day_type.csv` / `demand_day_type_errors.png` | Yearly demand errors within each fixed day type. Missing types in a year have no estimate, not zero error. |
| `monthly_day_type_shares.csv` | Monthly shares for the same days, to relate composition to error. |
| `mean_interval_widths.csv` | Mean interval widths across months, in percentage points. |
| `stage1_recheck.csv` | 2014 score reproduction for the four frozen algorithms. |
| `price_*_frozen.joblib` | Classifiers trained only on 2010–2013, for reuse in later experiments. |
| `daily_timing.csv` | Combined time for four models and baseline output construction; not a final sandbox benchmark. |
| `run_manifest.json` | Dates, versions, input/source hashes, bootstrap seed, boundary checks and unchanged-model checks. |
| `verification.json` | Independent checks of all labels/baselines, monthly/yearly metrics and an explicit whole-day resampling comparison. |

Blank cells in combined metric tables mean that a metric does not apply to that
task (e.g. demand accuracy), rather than failed predictions. Model predictions
are not clipped or replaced by baselines in this research run.

## Interval interpretation and remaining work

Each month uses 1,000 whole-day bootstrap replicates. Because every day has exactly
48 observations, averaging resampled daily mean errors is exactly equivalent to
resampling complete 48-value daily blocks. The IID approximation ignores serial
dependence; the day bootstrap preserves within-day dependence but still assumes
days are independent and exchangeable. Persistent weather and weekly patterns
can violate that assumption. Confidence intervals concern expected performance
under conditions like that month, not individual forecasts or live-week ranges.
Overlap of two months' intervals is not a formal test of their difference.

Day types are joined after the fact. Overall yearly MAPE is verified to equal
the sum of type shares times within-type MAPE. Both composition and within-type
error can change. Sparse late-period types must be interpreted using their day
counts, and associations do not establish causation or concept drift.

This run does not implement drift detectors, retraining, adapted-model
comparisons, live-week uncertainty or a final submission package. It provides
the frozen reference for those subsequent stages. Do not change initial model
choices or detector thresholds by looking ahead to simulation outcomes.
