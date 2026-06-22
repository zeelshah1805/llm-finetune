"""QLoRA fine-tuning for the resume -> JSON task (PLAN.md §3).

Trains a small LoRA adapter on a 4-bit base model so it fits a free Colab/Kaggle
T4. Designed to be run there:

    pip install -r requirements.txt
    python train.py --base Qwen/Qwen2.5-3B-Instruct --epochs 2

It loads data/{train,val}.jsonl, formats each example with the model's own chat
template (system + resume -> gold JSON completion), masks the prompt tokens so
loss is computed only on the completion, trains the adapter, logs train/eval
loss, saves the best checkpoint on eval loss, and writes a loss curve to
assets/loss_curve.png.

The frozen test set is NEVER touched here — that's the integrity rule.
"""
from __future__ import annotations

import argparse
import json
import os

from schema import SYSTEM_PROMPT, build_user_prompt, canonical_json

ROOT = os.path.dirname(os.path.abspath(__file__))


def load_split(name: str) -> list[dict]:
    path = os.path.join(ROOT, "data", f"{name}.jsonl")
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_chat_dataset(rows: list[dict], tokenizer, max_len: int):
    """Tokenize into input_ids/labels with the prompt portion masked (-100)."""
    from datasets import Dataset

    def encode(ex):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(ex["resume"])},
        ]
        prompt_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True)
        completion = canonical_json(ex["target"]) + tokenizer.eos_token
        completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]

        input_ids = (prompt_ids + completion_ids)[:max_len]
        labels = ([-100] * len(prompt_ids) + completion_ids)[:max_len]
        attn = [1] * len(input_ids)
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attn}

    ds = Dataset.from_list(rows)
    return ds.map(encode, remove_columns=ds.column_names)


class LossCurveLogger:
    """Collects loss points from the Trainer's log history for plotting."""

    def __init__(self):
        self.train, self.eval = [], []

    def collect(self, log_history):
        for h in log_history:
            if "loss" in h and "step" in h:
                self.train.append((h["step"], h["loss"]))
            if "eval_loss" in h and "step" in h:
                self.eval.append((h["step"], h["eval_loss"]))

    def plot(self, path: str):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as e:  # noqa: BLE001
            print(f"(skip loss curve: {e})")
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        if self.train:
            xs, ys = zip(*self.train)
            ax.plot(xs, ys, label="train loss")
        if self.eval:
            xs, ys = zip(*self.eval)
            ax.plot(xs, ys, marker="o", label="eval loss")
        ax.set_xlabel("step"); ax.set_ylabel("loss")
        ax.set_title("QLoRA training — resume → JSON"); ax.legend()
        fig.tight_layout(); fig.savefig(path, dpi=130)
        print(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--out", default=os.path.join(ROOT, "adapters", "qwen-resume"))
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--no-4bit", action="store_true")
    args = ap.parse_args()

    # Heavy deps imported here so `python train.py --help` works without them.
    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig, DataCollatorForSeq2Seq,
                              Trainer, TrainingArguments)
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = dict(trust_remote_code=True, device_map="auto",
                        torch_dtype=torch.bfloat16)
    if not args.no_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)

    model = AutoModelForCausalLM.from_pretrained(args.base, **model_kwargs)
    model.config.use_cache = False
    if not args.no_4bit:
        model = prepare_model_for_kbit_training(model)

    lora = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    train_ds = build_chat_dataset(load_split("train"), tokenizer, args.max_len)
    val_ds = build_chat_dataset(load_split("val"), tokenizer, args.max_len)

    collator = DataCollatorForSeq2Seq(tokenizer, label_pad_token_id=-100,
                                      padding=True)

    targs = TrainingArguments(
        output_dir=os.path.join(ROOT, "outputs"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=torch.cuda.is_available(),
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=20,
        save_strategy="steps",
        save_steps=20,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        gradient_checkpointing=True,
    )

    trainer = Trainer(model=model, args=targs, train_dataset=train_ds,
                      eval_dataset=val_ds, data_collator=collator)
    trainer.train()

    os.makedirs(args.out, exist_ok=True)
    trainer.save_model(args.out)          # saves the LoRA adapter
    tokenizer.save_pretrained(args.out)
    print(f"saved adapter -> {args.out}")

    logger = LossCurveLogger()
    logger.collect(trainer.state.log_history)
    logger.plot(os.path.join(ROOT, "assets", "loss_curve.png"))


if __name__ == "__main__":
    main()
