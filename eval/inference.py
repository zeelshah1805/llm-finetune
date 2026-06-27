"""Model loading + generation for evaluation.

Heavy deps (torch / transformers / peft) are imported lazily inside the loader
so the metrics and the rest of the pipeline import fine on a plain CPU/Windows
box. Actual generation needs a GPU (run on Colab/Kaggle).

A `ResumeParser` wraps a model + tokenizer and exposes `.parse(resume_text)`
returning (raw_text, latency_seconds, n_output_tokens). The eval harness treats
base and fine-tuned identically through this interface.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from schema import build_messages


@dataclass
class GenConfig:
    max_new_tokens: int = 384     # gold JSON fits well under this; caps base-model rambling
    temperature: float = 0.0      # greedy -> deterministic eval
    do_sample: bool = False


class ResumeParser:
    def __init__(self, model, tokenizer, gen: Optional[GenConfig] = None):
        self.model = model
        self.tok = tokenizer
        self.gen = gen or GenConfig()

    def _prompt(self, resume_text: str) -> str:
        return self.tok.apply_chat_template(
            build_messages(resume_text),
            tokenize=False,
            add_generation_prompt=True,
        )

    def parse(self, resume_text: str) -> tuple[str, float, int]:
        import torch  # lazy

        prompt = self._prompt(resume_text)
        inputs = self.tok(prompt, return_tensors="pt").to(self.model.device)
        in_len = inputs["input_ids"].shape[1]

        kwargs = dict(
            max_new_tokens=self.gen.max_new_tokens,
            do_sample=self.gen.do_sample,
            pad_token_id=self.tok.eos_token_id,
        )
        if self.gen.do_sample:
            kwargs["temperature"] = self.gen.temperature

        t0 = time.perf_counter()
        with torch.no_grad():
            out = self.model.generate(**inputs, **kwargs)
        latency = time.perf_counter() - t0

        new_tokens = out[0][in_len:]
        text = self.tok.decode(new_tokens, skip_special_tokens=True)
        return text, latency, int(new_tokens.shape[0])


def _bnb_config():
    from transformers import BitsAndBytesConfig
    import torch
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def load_parser(base_model: str, adapter_path: Optional[str] = None,
                load_4bit: bool = True, gen: Optional[GenConfig] = None) -> ResumeParser:
    """Load base (and optionally apply a LoRA adapter) as a ResumeParser.

    base_model : HF id, e.g. "Qwen/Qwen2.5-3B-Instruct"
    adapter_path : local dir or HF id of a PEFT adapter; None = plain base model.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model_kwargs = dict(
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    if load_4bit:
        model_kwargs["quantization_config"] = _bnb_config()

    model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)

    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return ResumeParser(model, tok, gen)
