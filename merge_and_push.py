"""Merge the LoRA adapter into the base model and (optionally) push to HF Hub.

Run after training (PLAN.md §0 deliverable: merged model + model card on Hub):

    python merge_and_push.py --base Qwen/Qwen2.5-3B-Instruct \
                             --adapter adapters/qwen-resume \
                             --out merged_model \
                             --push your-username/qwen2.5-3b-resume-json

Merge in fp16 (not 4-bit) so the published weights are usable everywhere.
"""
from __future__ import annotations

import argparse
import os


MODEL_CARD = """---
license: apache-2.0
base_model: {base}
tags: [lora, qlora, resume-parsing, json, information-extraction]
---

# {repo} — Resume → JSON (QLoRA fine-tune)

Fine-tune of `{base}` that converts free-text resumes into a fixed JSON schema
(name, email, phone, total_years_experience, skills, education, work_experience).

- **Method:** QLoRA (4-bit base, LoRA r=16, alpha=32) via `trl`/`peft`.
- **Eval:** benchmarked base vs. fine-tuned on a frozen, version-controlled test
  set with field-level F1, JSON-validity, and hallucinated-field rate.
- See the project repo for the data pipeline, training script, and the full
  evaluation framework that produced the comparison table.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("{repo}")
model = AutoModelForCausalLM.from_pretrained("{repo}", device_map="auto")
```
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--adapter", default="adapters/qwen-resume")
    ap.add_argument("--out", default="merged_model")
    ap.add_argument("--push", help="HF Hub repo id to push to (optional)")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print(f"loading base {args.base} (fp16) ...")
    base = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.float16, device_map="auto",
        trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)

    print(f"applying adapter {args.adapter} and merging ...")
    merged = PeftModel.from_pretrained(base, args.adapter).merge_and_unload()

    os.makedirs(args.out, exist_ok=True)
    merged.save_pretrained(args.out, safe_serialization=True)
    tok.save_pretrained(args.out)
    print(f"merged model -> {args.out}")

    if args.push:
        card = MODEL_CARD.format(base=args.base, repo=args.push)
        with open(os.path.join(args.out, "README.md"), "w", encoding="utf-8") as f:
            f.write(card)
        print(f"pushing to https://huggingface.co/{args.push} ...")
        merged.push_to_hub(args.push)
        tok.push_to_hub(args.push)
        print("done. (ensure you ran `huggingface-cli login` first)")


if __name__ == "__main__":
    main()
