# A/B integration index

- [A demand handoff](A_demand/DEMAND_HANDOFF.md): Ridge/MLP settings, validation, diagnostics and frozen-model interface.
- [A original instructions](A_demand/README.md) and [worksheet](A_demand/REPORT_WORKSHEET.md).
- [B price handoff](B_price/PRICE_HANDOFF.md): Logistic/HistGradientBoosting settings and validation.
- [B supplied report](B_price/Price_Classification_Report.docx) is preserved as received, without editing or endorsing its authorship.

Models and selected configuration are integrated into `experiments/`; delivered Stage 1 evidence is in `outputs/stage1/`; demand frozen artifacts are in `artifacts/`. Shared source files have been compared byte-for-byte and preserved.

Integration changes only to A's ancillary tools: generated documents and environment records go under `handoffs/A_demand/` instead of overwriting the course README; handoff packaging includes the documentation directory. No model training or selected settings were changed. Original source/model hashes remain in the supplied manifest.

The original local handoff folders are preserved. Pilot runs, duplicate reproduction outputs, .DS_Store, caches, Python environments and released training data are not included in this integration.
