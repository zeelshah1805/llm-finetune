"""Run the base-vs-fine-tuned evaluation and emit the comparison artifacts.

Three modes:

  # 1. Real eval: load models and generate on the frozen test set (needs GPU).
  python eval/run_eval.py --base Qwen/Qwen2.5-3B-Instruct \
                          --adapter adapters/qwen-resume

  # 2. Offline scoring: you generated predictions on Colab, downloaded the
  #    *.jsonl, and just want the table/chart locally (no GPU needed).
  python eval/run_eval.py --from-predictions results/raw/base.jsonl:Base \
                          results/raw/finetuned.jsonl:"Fine-tuned"

  # 3. Demo: simulate a realistic base/fine-tuned gap from the gold test set so
  #    the full reporting pipeline runs anywhere. Clearly labelled as simulated.
  python eval/run_eval.py --demo

Outputs:
  results/comparison.md     the headline base-vs-fine-tuned table
  results/charts/*.png      bar charts (field F1, format-failure rate)
  results/raw/<label>.jsonl raw predictions (mode 1)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from schema import TOP_LEVEL_KEYS  # noqa: E402
from eval.metrics import aggregate, extract_json, score_example  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def load_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Scoring a set of predictions
# ---------------------------------------------------------------------------


def score_predictions(rows: list[dict]) -> dict:
    """rows: [{target, raw, [latency], [n_tokens]}]. `raw` is model text or a
    pre-parsed dict; gold is `target`."""
    per_example, lat, toks = [], [], []
    for r in rows:
        raw = r.get("raw")
        pred_obj = raw if isinstance(raw, dict) else extract_json(raw or "")
        per_example.append(score_example(pred_obj, r["target"]))
        if r.get("latency") is not None:
            lat.append(r["latency"])
        if r.get("n_tokens") is not None:
            toks.append(r["n_tokens"])
    return aggregate(per_example, latencies=lat or None, out_tokens=toks or None)


# ---------------------------------------------------------------------------
# Mode 1: real generation
# ---------------------------------------------------------------------------


def run_model(label: str, base: str, adapter, test: list[dict], load_4bit: bool,
              max_new_tokens: int = 384) -> list[dict]:
    from eval.inference import GenConfig, load_parser  # lazy: heavy deps

    print(f"[{label}] loading base={base} adapter={adapter} 4bit={load_4bit}")
    parser = load_parser(base, adapter_path=adapter, load_4bit=load_4bit,
                         gen=GenConfig(max_new_tokens=max_new_tokens))

    rows = []
    for i, ex in enumerate(test, 1):
        raw, latency, n_tok = parser.parse(ex["resume"])
        rows.append({"resume": ex["resume"], "target": ex["target"], "raw": raw,
                     "latency": latency, "n_tokens": n_tok})
        if i % 10 == 0 or i == len(test):
            print(f"  [{label}] {i}/{len(test)}")
    save_jsonl(os.path.join(RESULTS, "raw", f"{label}.jsonl"), rows)
    return rows


# ---------------------------------------------------------------------------
# Mode 3: demo / simulated predictions (no model)
# ---------------------------------------------------------------------------


def _corrupt(target: dict, rng: random.Random, bad: float) -> dict | str:
    """Simulate an *untuned* base model: sometimes wraps in prose / fences,
    invents an extra key, drops contact fields, mangles a skill."""
    obj = json.loads(json.dumps(target))
    if rng.random() < bad:                       # invent a field
        obj["summary"] = "Experienced professional."
    if rng.random() < bad and obj["skills"]:     # mangle a skill
        obj["skills"][0] = obj["skills"][0] + "3"
    if rng.random() < bad * 0.6:                 # stringify a number
        obj["total_years_experience"] = str(obj["total_years_experience"])
    if rng.random() < bad * 0.5 and obj.get("email"):  # hallucinate vs drop
        obj["email"] = None

    s = json.dumps(obj, indent=2)
    roll = rng.random()
    if roll < bad * 0.5:                          # wrap in a fence + prose
        return f"Sure! Here is the parsed resume:\n```json\n{s}\n```"
    if roll < bad * 0.65:                         # truncated / invalid json
        return s[: int(len(s) * 0.7)]
    return s


def _polish(target: dict, rng: random.Random, slip: float) -> str:
    """Simulate the fine-tuned model: clean JSON, occasional tiny slip."""
    obj = json.loads(json.dumps(target))
    if rng.random() < slip and obj["skills"]:
        obj["skills"] = obj["skills"][:-1]       # miss one skill rarely
    return json.dumps(obj)


def demo_predictions(test: list[dict], seed: int = 7) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    base_rows, ft_rows = [], []
    for ex in test:
        base_rows.append({
            "resume": ex["resume"],
            "target": ex["target"],
            "raw": _corrupt(ex["target"], rng, bad=0.45),
            "latency": rng.uniform(1.6, 2.4), "n_tokens": rng.randint(160, 240),
        })
        ft_rows.append({
            "resume": ex["resume"],
            "target": ex["target"],
            "raw": _polish(ex["target"], rng, slip=0.06),
            "latency": rng.uniform(1.1, 1.6), "n_tokens": rng.randint(120, 170),
        })
    return {"Base (Qwen2.5-3B)": base_rows, "Fine-tuned (QLoRA)": ft_rows}


# ---------------------------------------------------------------------------
# Reporting: comparison.md + charts
# ---------------------------------------------------------------------------

_ROWS = [
    ("JSON validity %", "json_validity", "pct", "up"),
    ("Schema conformance %", "schema_conformance", "pct", "up"),
    ("Field-level F1", "field_f1", "f3", "up"),
    ("Hallucinated-field rate %", "hallucinated_field_rate", "pct", "down"),
    ("Format-failure rate %", "format_failure_rate", "pct", "down"),
    ("Avg latency (s)", "avg_latency_s", "f2", "down"),
    ("Avg output tokens", "avg_output_tokens", "f0", "down"),
    ("LLM-judge mean (1–5)", "judge_mean", "f2", "up"),
    ("LLM-judge %=5", "judge_pct_5", "pct", "up"),
]


def _fmt(val, kind: str) -> str:
    if val is None:
        return "—"
    if kind == "pct":
        return f"{val * 100:.0f}%"
    if kind == "f3":
        return f"{val:.3f}"
    if kind == "f2":
        return f"{val:.2f}"
    if kind == "f0":
        return f"{val:.0f}"
    return str(val)


def _delta(base, ft, kind: str, better: str) -> str:
    if base is None or ft is None:
        return "—"
    d = ft - base
    if kind == "pct":
        s = f"{d * 100:+.0f}"
    elif kind == "f3":
        s = f"{d:+.3f}"
    elif kind == "f2":
        s = f"{d:+.2f}"
    else:
        s = f"{d:+.0f}"
    improved = (d > 0 and better == "up") or (d < 0 and better == "down")
    mark = " ✅" if improved and abs(d) > 1e-9 else (" ⚠️" if abs(d) > 1e-9 else "")
    return s + mark


def write_comparison_md(results: dict[str, dict], path: str, demo: bool) -> None:
    labels = list(results)
    base_l, ft_l = labels[0], labels[-1]
    base, ft = results[base_l], results[ft_l]

    lines = ["# Base vs. Fine-tuned — Resume → JSON\n"]
    if demo:
        lines.append("> ⚠️ **Simulated numbers** (`--demo`). Replace by running the "
                     "real eval on a GPU. Shown to demonstrate the reporting pipeline.\n")
    lines.append(f"Frozen test set: **{base.get('n', '?')}** examples. "
                 "Headline metric: field-level F1.\n")

    lines.append(f"| Metric | {base_l} | {ft_l} | Δ |")
    lines.append("|---|---|---|---|")
    for name, key, kind, better in _ROWS:
        # skip a metric entirely if no model reports it (e.g. judge not run)
        if all(results[l].get(key) is None for l in labels):
            continue
        bv, fv = base.get(key), ft.get(key)
        lines.append(f"| {name} | {_fmt(bv, kind)} | {_fmt(fv, kind)} | "
                     f"{_delta(bv, fv, kind, better)} |")

    lines.append("\n## Per-field F1\n")
    fields = list(base.get("per_field_f1", {}))
    lines.append("| Field | " + " | ".join(labels) + " |")
    lines.append("|" + "---|" * (len(labels) + 1))
    for fld in fields:
        cells = [f"{results[l]['per_field_f1'][fld]:.3f}" for l in labels]
        lines.append(f"| {fld} | " + " | ".join(cells) + " |")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {path}")


def write_charts(results: dict[str, dict], outdir: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"(skipping charts: matplotlib unavailable: {e})")
        return
    os.makedirs(outdir, exist_ok=True)
    labels = list(results)

    # chart 1: per-field F1 grouped bars
    fields = list(results[labels[0]].get("per_field_f1", {}))
    if fields:
        x = range(len(fields))
        width = 0.8 / len(labels)
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, l in enumerate(labels):
            vals = [results[l]["per_field_f1"][f] for f in fields]
            ax.bar([xi + i * width for xi in x], vals, width, label=l)
        ax.set_xticks([xi + width * (len(labels) - 1) / 2 for xi in x])
        ax.set_xticklabels(fields, rotation=30, ha="right")
        ax.set_ylabel("F1"); ax.set_ylim(0, 1.05)
        ax.set_title("Per-field F1: base vs. fine-tuned"); ax.legend()
        fig.tight_layout()
        p = os.path.join(outdir, "per_field_f1.png")
        fig.savefig(p, dpi=130); plt.close(fig)
        print(f"wrote {p}")

    # chart 2: headline metrics
    headline = [("Field F1", "field_f1"), ("JSON validity", "json_validity"),
                ("Schema OK", "schema_conformance"),
                ("Format-fail", "format_failure_rate")]
    x = range(len(headline))
    width = 0.8 / len(labels)
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, l in enumerate(labels):
        vals = [results[l].get(k, 0) or 0 for _, k in headline]
        ax.bar([xi + i * width for xi in x], vals, width, label=l)
    ax.set_xticks([xi + width * (len(labels) - 1) / 2 for xi in x])
    ax.set_xticklabels([n for n, _ in headline])
    ax.set_ylabel("score (0–1)"); ax.set_ylim(0, 1.05)
    ax.set_title("Headline metrics: base vs. fine-tuned"); ax.legend()
    fig.tight_layout()
    p = os.path.join(outdir, "headline.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    print(f"wrote {p}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_pred_arg(spec: str) -> tuple[str, str]:
    """'path/file.jsonl:Label' -> (path, label). Label optional."""
    if ":" in spec and not spec[1:3] == ":\\":  # avoid splitting Windows drive
        path, label = spec.rsplit(":", 1)
        return path, label
    return spec, os.path.splitext(os.path.basename(spec))[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test", default=os.path.join(ROOT, "data", "test.jsonl"))
    ap.add_argument("--base", help="HF base model id (mode 1)")
    ap.add_argument("--adapter", help="LoRA adapter path/id (mode 1)")
    ap.add_argument("--no-4bit", action="store_true", help="disable 4-bit load")
    ap.add_argument("--from-predictions", nargs="+", metavar="FILE[:LABEL]",
                    help="score saved prediction jsonl files (mode 2)")
    ap.add_argument("--demo", action="store_true", help="simulated eval (mode 3)")
    ap.add_argument("--judge", action="store_true",
                    help="add a local Ollama LLM-as-judge score (secondary signal)")
    ap.add_argument("--judge-model", default=None,
                    help="Ollama model for the judge (default: llama3.1:8b)")
    ap.add_argument("--limit", type=int, default=None,
                    help="evaluate only the first N test examples (quick smoke test)")
    ap.add_argument("--max-new-tokens", type=int, default=384,
                    help="generation cap per example (default 384)")
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    pred_rows: dict[str, list[dict]] = {}
    demo = False

    if args.from_predictions:
        for spec in args.from_predictions:
            path, label = _parse_pred_arg(spec)
            pred_rows[label] = load_jsonl(path)

    elif args.demo:
        demo = True
        test = load_jsonl(args.test)
        pred_rows = demo_predictions(test)

    elif args.base:
        test = load_jsonl(args.test)
        if args.limit:
            test = test[:args.limit]
            print(f"(smoke test: evaluating only {len(test)} examples)")
        pred_rows["Base (Qwen2.5-3B)"] = run_model(
            "base", args.base, None, test, not args.no_4bit, args.max_new_tokens)
        if args.adapter:
            pred_rows["Fine-tuned (QLoRA)"] = run_model(
                "finetuned", args.base, args.adapter, test, not args.no_4bit,
                args.max_new_tokens)
    else:
        ap.error("choose a mode: --demo, --from-predictions, or --base [--adapter]")

    results: dict[str, dict] = {label: score_predictions(rows)
                                for label, rows in pred_rows.items()}

    if args.judge:
        from eval.judge import DEFAULT_MODEL, judge_predictions
        model = args.judge_model or DEFAULT_MODEL
        for label, rows in pred_rows.items():
            print(f"[judge] scoring {label} with {model} ...")
            stats = judge_predictions(rows, model=model)
            if stats:
                results[label].update(stats)

    write_comparison_md(results, os.path.join(RESULTS, "comparison.md"), demo)
    write_charts(results, os.path.join(RESULTS, "charts"))
    print("\nSummary:")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
