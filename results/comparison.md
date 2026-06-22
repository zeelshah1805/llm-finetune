# Base vs. Fine-tuned — Resume → JSON

> ⚠️ **Simulated numbers** (`--demo`). Replace by running the real eval on a GPU. Shown to demonstrate the reporting pipeline.

Frozen test set: **40** examples. Headline metric: field-level F1.

| Metric | Base (Qwen2.5-3B) | Fine-tuned (QLoRA) | Δ |
|---|---|---|---|
| JSON validity % | 100% | 100% | +0 |
| Schema conformance % | 57% | 100% | +43 ✅ |
| Field-level F1 | 0.941 | 1.000 | +0.059 ✅ |
| Hallucinated-field rate % | 42% | 0% | -42 ✅ |
| Format-failure rate % | 42% | 0% | -42 ✅ |
| Avg latency (s) | 2.00 | 1.33 | -0.67 ✅ |
| Avg output tokens | 198 | 140 | -58 ✅ |

## Per-field F1

| Field | Base (Qwen2.5-3B) | Fine-tuned (QLoRA) |
|---|---|---|
| name | 1.000 | 1.000 |
| email | 0.725 | 1.000 |
| phone | 1.000 | 1.000 |
| total_years_experience | 1.000 | 1.000 |
| skills | 0.905 | 1.000 |
| education | 1.000 | 1.000 |
| work_experience | 1.000 | 1.000 |
