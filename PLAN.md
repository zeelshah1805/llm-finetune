# Project 3 — LLM Fine-tuning + Evaluation Framework

**One-line pitch:** LoRA/QLoRA fine-tune a small open-source model on a domain task, then *prove it got better* with a custom evaluation framework that benchmarks base vs. fine-tuned on real metrics.

**Why this project wins interviews:** Most candidates have *used* an LLM. Very few have *changed the weights* and then *measured the change rigorously*. This project shows both — and the eval framework is the part senior interviewers actually care about, because it proves you know fine-tuning can make a model worse and you have the discipline to catch it.

**Stack is 100% free / OSS / no paid GPU:** Google Colab (free T4) or Kaggle (free T4×2 / 30h per week), Hugging Face `transformers` + `peft` + `trl` + `bitsandbytes`, `datasets`, base model from Hugging Face Hub. No OpenAI key required. Optional local LLM-as-judge via Ollama.

---

## §0 — Goal & success criteria

**Goal:** Take a small open-source base model (3B–7B), fine-tune it with QLoRA on one well-defined task, and deliver a repo where anyone can run `eval` and see a base-vs-finetuned comparison table.

**Done = all of these are true:**

- QLoRA adapter trains end-to-end on a free Colab/Kaggle T4 without OOM.
- A held-out test set the model never saw during training exists and is version-controlled.
- The eval framework outputs a single comparison table: **base vs. fine-tuned** across ≥3 metrics.
- Fine-tuned model beats base on the target metric **and** you can explain any metric where it got worse.
- README has the comparison table, training curves, and a "how to reproduce" section.
- Merged model (or adapter) pushed to Hugging Face Hub with a model card.

**Explicit non-goals:** No RLHF/DPO in v1 (note it as a future extension). No multi-GPU distributed training. No serving infra (that's Project 4, the observability dashboard).

---

## §1 — Pick the task and base model (do this first, it shapes everything)

The single most important decision. A *narrow, structured* task makes fine-tuning visibly work and makes evaluation objective. Avoid open-ended chat — it's hard to measure and base models are already good at it.

**Recommended task options (pick ONE):**

| Task | Why it's good | Metric that proves success |
|---|---|---|
| **Resume → structured JSON** (skills, years, roles) | Ties to your SixSigma recruitment work; objective output | JSON-validity %, field-level exact match / F1 |
| **Indian legal clause classification** | Niche, impressive, public datasets exist | Accuracy, per-class F1 |
| **Support ticket → category + priority** | Clean labels, business-relevant | Accuracy, macro-F1 |
| **Text → SQL (single table)** | Very interview-relevant | Execution-match accuracy |

> Pick the **resume → JSON** task unless you have a strong reason. It connects directly to your production experience, the output is structured (easy to score automatically), and it tells a clean story: "my company does recruitment automation; I fine-tuned a model to do the parsing step better and cheaper than an API call."

**Base model candidates (all free, all fit QLoRA on a T4):**

- `Qwen/Qwen2.5-3B-Instruct` — strong, small, T4-friendly (recommended default)
- `meta-llama/Llama-3.2-3B-Instruct` — recognizable name, gated (accept license on HF)
- `mistralai/Mistral-7B-Instruct-v0.3` — 7B; works with QLoRA on T4 but tighter on memory

Start with **Qwen2.5-3B-Instruct**: it trains fast, leaves memory headroom, and removes "I couldn't get it to run" risk.

---

## §2 — Data pipeline

Fine-tuning is 80% data work. Treat the dataset as the deliverable.

1. **Source the data.** Use a public dataset (Hugging Face Hub) or generate a synthetic set. For resume→JSON you can synthesize: write 150–400 varied resumes (templated + LLM-augmented) paired with gold JSON.
2. **Define the schema / label space precisely.** One JSON schema, fixed keys, documented in the repo. Ambiguity here destroys eval reliability.
3. **Format into instruction style.** Build the chat-template prompt: a system instruction, the input, and the target completion. Use the base model's *exact* chat template (`tokenizer.apply_chat_template`).
4. **Split cleanly.** Train / validation / test (e.g., 80/10/10). **The test set must be frozen and never seen during training or prompt tuning.** This is the integrity backbone of the whole project — say this in interviews.
5. **Version it.** Commit the splits (or a deterministic seed + script). Reproducibility is a selling point.

**Deliverable:** `data/` with `train.jsonl`, `val.jsonl`, `test.jsonl` and a `build_dataset.py` that regenerates them deterministically.

---

## §3 — Training (QLoRA)

The goal is a small adapter, not a full model retrain — that's what makes it run on free hardware.

**Config that works on a free T4:**

- Load base in **4-bit** (`bitsandbytes`, nf4, double quant).
- **LoRA** via `peft`: `r=16`, `lora_alpha=32`, `lora_dropout=0.05`, target the attention + MLP projection layers.
- Train with `trl`'s `SFTTrainer`: batch size 1–2 + gradient accumulation 8–16, `bf16`/`fp16`, learning rate `2e-4`, cosine schedule, 1–3 epochs.
- Log `train/loss` and `eval/loss` every N steps; save the best checkpoint on val loss.

**Guardrails:**

- Watch for **overfitting** (train loss drops, val loss rises) — small datasets overfit fast. Early-stop.
- Save **training curves** (a loss plot) — this goes in the README.
- Keep the adapter small; merge to a full model only at the end for the Hub upload.

**Deliverable:** `train.py` (or a clean Colab notebook), saved adapter, and `assets/loss_curve.png`.

---

## §4 — The Evaluation Framework (the real centerpiece)

This is what separates you from "I followed a fine-tuning tutorial." Build it as a standalone, reusable module that takes a model + the frozen test set and emits a metrics table.

**Run every metric on BOTH models** (base, fine-tuned) over the *same* frozen test set:

1. **Task accuracy metric** (the headline number) — choose to fit the task:
   - JSON tasks → JSON-validity rate + field-level exact-match / F1.
   - Classification → accuracy + macro-F1 + confusion matrix.
   - Text→SQL → execution-match accuracy.
2. **Hallucination / format-failure rate** — % of outputs that are invalid (bad JSON, wrong label not in schema, hallucinated fields). Fine-tuning should *crush* this vs. base.
3. **Optional quality score via LLM-as-judge** — a rubric scored 1–5 by a judge model. Run the judge locally with **Ollama** (free) so no API key is needed; document the rubric and that scores are relative.
4. **Efficiency** — tokens/output and latency, to argue the cost case ("smaller fine-tuned model matches a bigger base at lower cost").

**Output:** one `results/comparison.md` table like:

| Metric | Base (Qwen2.5-3B) | Fine-tuned | Δ |
|---|---|---|---|
| JSON validity % | 71% | 98% | +27 |
| Field F1 | 0.62 | 0.89 | +0.27 |
| Hallucinated fields % | 14% | 2% | −12 |
| Avg latency (s) | … | … | … |

**Deliverable:** `eval/` module + `run_eval.py` that produces `results/comparison.md` and a bar chart. **This table is your strongest interview artifact** — it's concrete proof, with numbers, that your change helped.

---

## §5 — Build phases (suggested 2–3 week schedule)

- **Phase 1 — Scaffold & baseline (days 1–2):** repo, env, pick task + base model, load base, run it on 10 test examples, eyeball failures. Commit a naive baseline.
- **Phase 2 — Data (days 3–6):** build/clean dataset, define schema, format with chat template, freeze splits, `build_dataset.py`.
- **Phase 3 — Training (days 7–10):** QLoRA config, get it running on Colab/Kaggle without OOM, tune epochs, save adapter + loss curve.
- **Phase 4 — Eval framework (days 11–14):** build metrics module, run base vs. fine-tuned, generate `comparison.md` + charts.
- **Phase 5 — Ablation + polish (days 15–18):** vary one thing (epochs, LoRA rank, or dataset size) to show its effect; write README; push model + card to HF Hub; record a 60-sec demo GIF.

Ship after Phase 4 if time-constrained; Phase 5 is what makes it portfolio-grade.

---

## §6 — Repo structure

```
llm-finetune-eval/
├── README.md                # table + curves + reproduce steps (the storefront)
├── data/
│   ├── build_dataset.py
│   ├── train.jsonl / val.jsonl / test.jsonl
│   └── schema.md
├── train.py                 # or notebooks/train.ipynb
├── eval/
│   ├── metrics.py           # task metric, hallucination rate, judge
│   └── run_eval.py
├── results/
│   ├── comparison.md        # base vs fine-tuned table
│   └── charts/
├── assets/loss_curve.png
└── requirements.txt
```

---

## §7 — Risks & how to defuse them

- **OOM on free GPU** → use 3B model + 4-bit + batch size 1 + grad accumulation; Qwen2.5-3B is the safe default. Kaggle T4×2 as backup.
- **Fine-tuned model gets *worse*** → this is normal and is *good interview material*; show you caught it via the eval table and explain why (too few epochs, bad data, catastrophic forgetting). Never hide a regression.
- **Tiny dataset overfits** → early-stop on val loss, augment data, keep epochs low (1–2).
- **Unreliable metrics** → freeze the test set, fix the schema, prefer objective metrics over LLM-judge as the headline.
- **Gated models block you** → Qwen is open; use it to avoid Llama license friction.
- **"Did it really improve or did you just test on training data?"** → the frozen, version-controlled test set is your answer. Lead with it.

---

## §8 — Portfolio & interview impact

**Resume bullet:**
> Fine-tuned Qwen2.5-3B with QLoRA for structured resume parsing, building a custom evaluation framework (field-level F1, hallucination rate, LLM-as-judge) that benchmarked base vs. fine-tuned on a frozen test set — raising JSON validity from 71% → 98% and field F1 from 0.62 → 0.89.

**Defensible interview answers this project earns you:**

- *"Why LoRA/QLoRA instead of full fine-tuning?"* — Full fine-tuning of a 7B updates billions of params and needs many GB of GPU memory; QLoRA freezes the 4-bit base and trains small low-rank adapters (<1% of params), so it fits on a free T4 while reaching comparable task quality.
- *"How did you know it actually improved?"* — Frozen, version-controlled test set the model never saw; same eval run on base and fine-tuned; objective metrics first, LLM-judge only as a secondary signal.
- *"How do you prevent overfitting on a small dataset?"* — Early stopping on val loss, low epochs, data augmentation, watching the train/val gap on the loss curve.
- *"What's your hallucination metric?"* — % of outputs that violate the schema (invalid JSON, out-of-vocab labels, invented fields); fine-tuning drove it from 14% → 2%.
- *"When would you NOT fine-tune?"* — When prompting/RAG already meets the bar, when data is scarce or noisy, or when the task changes often — fine-tuning bakes in behavior and is costly to re-do.

**Strongest single artifact:** the `results/comparison.md` base-vs-fine-tuned table. Pin the repo, lead the README with that table, and link it as the third project on your portfolio site (the "LLM Fine-tuning + Eval Framework" bento card).

**Sequencing note:** This is Project 3 of 4. Project 1 was the Multi-Agent Research System, Project 2 the Advanced RAG Pipeline, and Project 4 is the AI Observability Dashboard. Build at least 1 and 2 first so the GitHub links on your portfolio aren't empty.
