# LLM Fine-tuning + Evaluation Framework — Resume → JSON

QLoRA fine-tune of a small open-source model (**Qwen2.5-3B-Instruct**) on a
structured task — turning free-text resumes into a fixed JSON schema — and a
**custom evaluation framework that proves the change helped** by benchmarking
base vs. fine-tuned on a frozen, version-controlled test set.

> The eval framework is the centerpiece. Anyone can `python eval/run_eval.py`
> and get a base-vs-fine-tuned comparison table with field-level F1, JSON
> validity, and hallucinated-field rate — concrete, reproducible numbers.

---

## Headline result

<!-- Replace with real numbers after running the GPU eval (results/comparison.md). -->
The table below is produced by the eval framework. Numbers shown are from
`--demo` (simulated) until the real training run is done; the pipeline that
generates them is real.

| Metric | Base (Qwen2.5-3B) | Fine-tuned (QLoRA) | Δ |
|---|---|---|---|
| JSON validity % | 100% | 100% | +0 |
| Schema conformance % | 57% | 100% | +43 ✅ |
| Field-level F1 | 0.941 | 1.000 | +0.059 ✅ |
| Hallucinated-field rate % | 42% | 0% | −42 ✅ |
| Format-failure rate % | 42% | 0% | −42 ✅ |

![Headline metrics](results/charts/headline.png)
![Per-field F1](results/charts/per_field_f1.png)

---

## Why this project

Most candidates have *used* an LLM. Few have *changed the weights* and then
*measured the change rigorously*. This repo shows both — and the eval framework
proves discipline: it can catch a fine-tune that made the model **worse**, which
is the failure mode senior interviewers probe for.

- **QLoRA** so it trains end-to-end on a **free Colab/Kaggle T4** (no paid GPU).
- A **frozen test set** the model never sees during training or prompt tuning —
  the integrity backbone of the whole comparison.
- Objective, automatic metrics first; LLM-as-judge only as a secondary signal.

---

## Repo layout

```
.
├── schema.py             # single source of truth: JSON schema + prompt + chat formatting
├── data/
│   ├── build_dataset.py  # deterministic synthetic resume→JSON generator
│   ├── schema.md         # human-readable schema + how it maps to metrics
│   └── train/val/test.jsonl  # frozen splits (80/10/10), version-controlled
├── train.py              # QLoRA fine-tuning (4-bit base, LoRA adapter)
├── merge_and_push.py     # merge adapter → fp16 model + push to HF Hub
├── notebooks/
│   └── train_and_eval.ipynb  # one-click Colab/Kaggle: install→train→eval→table
├── eval/
│   ├── metrics.py        # JSON validity, schema conformance, field-level F1, hallucination
│   ├── inference.py      # model loading + generation (base or base+adapter)
│   └── run_eval.py       # orchestrates base-vs-fine-tuned → comparison.md + charts
├── results/
│   ├── comparison.md     # the base-vs-fine-tuned table (generated)
│   └── charts/           # bar charts (generated)
└── requirements.txt
```

---

## The task & schema

Input: a resume in any of several free-text layouts (classic, narrative,
bullets, terse). Output: one JSON object with fixed keys:

```json
{
  "name": "string",
  "email": "string | null",
  "phone": "string | null",
  "total_years_experience": 5,
  "skills": ["Python", "SQL"],
  "education": [{"degree": "B.Tech in CS", "institution": "IIT Bombay", "year": 2018}],
  "work_experience": [{"title": "ML Engineer", "company": "Acme", "years": 3}]
}
```

Full contract and metric mapping: [data/schema.md](data/schema.md).

---

## Reproduce

### 0. Setup

```bash
pip install -r requirements.txt
```

(The eval + data tooling runs on plain CPU/Windows. Training needs a CUDA GPU —
use Colab/Kaggle, where you also `pip install bitsandbytes`.)

> **Fastest path:** open [`notebooks/train_and_eval.ipynb`](notebooks/train_and_eval.ipynb)
> in Colab (set runtime to T4 GPU) and run all cells — it installs, trains,
> evaluates, and renders the comparison table for you.

### 1. Build the dataset (deterministic)

```bash
python data/build_dataset.py --n 400 --seed 42
# → data/train.jsonl (320), val.jsonl (40), test.jsonl (40)
```

The splits are committed, so the test set is frozen and identical for everyone.

### 2. Fine-tune (on a free T4)

```bash
python train.py --base Qwen/Qwen2.5-3B-Instruct --epochs 2
# → adapters/qwen-resume/  and  assets/loss_curve.png
```

QLoRA config: 4-bit nf4 base, LoRA `r=16`, `alpha=32`, `dropout=0.05` on the
attention + MLP projections; batch 2 × grad-accum 8, lr 2e-4 cosine, best
checkpoint on eval loss. Watch the train/eval loss gap for overfitting.

### 3. Evaluate base vs. fine-tuned

```bash
# real eval (GPU): generates predictions for both models, scores, writes table+charts
python eval/run_eval.py --base Qwen/Qwen2.5-3B-Instruct --adapter adapters/qwen-resume

# offline: score prediction files you generated on Colab, no GPU needed
python eval/run_eval.py --from-predictions results/raw/base.jsonl:Base \
                        results/raw/finetuned.jsonl:"Fine-tuned"

# demo: simulate the gap to show the reporting pipeline anywhere
python eval/run_eval.py --demo
```

→ `results/comparison.md` + `results/charts/*.png`.

### 4. Publish (optional)

```bash
python merge_and_push.py --adapter adapters/qwen-resume \
                         --push your-username/qwen2.5-3b-resume-json
```

---

## Evaluation framework details

Every metric runs on **both** models over the **same** frozen test set:

1. **Task accuracy** — field-level **F1** (micro-averaged over atoms: each
   scalar field, each skill, each education/work item). The headline number.
2. **Format-failure** — JSON-validity rate, schema conformance, and
   **hallucinated-field rate** (extra/invented keys). Fine-tuning should crush
   these vs. base.
3. **Efficiency** — average latency and output tokens, for the cost argument.

Metrics are pure functions (no torch) with a built-in self-test:

```bash
python eval/metrics.py   # runs assertions + prints a sample aggregate
```

### LLM-as-judge (optional, local Ollama — no API key)

A secondary, *relative* quality signal. A local judge model scores each
predicted JSON 1–5 against the resume + gold on a documented rubric
([eval/judge.py](eval/judge.py)). Objective field-F1 stays the headline; the
judge is a sanity cross-check.

```bash
# one-time: install Ollama (https://ollama.com), then
ollama pull llama3.1:8b
# add the judge to any eval run:
python eval/run_eval.py --demo --judge
python eval/run_eval.py --base Qwen/Qwen2.5-3B-Instruct --adapter adapters/qwen-resume --judge
```

If no Ollama server is reachable, judging is skipped gracefully and the rest of
the eval is unaffected — so CI / GPU-less runs never break.

---

## Defensible interview answers this earns

- **Why QLoRA vs. full fine-tuning?** Full FT of a 7B updates billions of params
  and needs many GB of GPU memory; QLoRA freezes the 4-bit base and trains
  low-rank adapters (<1% of params), fitting a free T4 at comparable quality.
- **How do you know it improved?** Frozen, version-controlled test set never seen
  in training; identical eval on base and fine-tuned; objective metrics first.
- **How do you prevent overfitting?** Early stop on eval loss, low epochs, watch
  the train/eval gap on the loss curve, augment data.
- **What's the hallucination metric?** % of outputs that violate the schema
  (invalid JSON, out-of-schema keys, invented fields).
- **When would you NOT fine-tune?** When prompting/RAG already meets the bar,
  data is scarce/noisy, or the task changes often.

---

## Non-goals (v1)

No RLHF/DPO (future extension), no multi-GPU distributed training, no serving
infra. Scope is: fine-tune one structured task and **prove the gain rigorously**.
