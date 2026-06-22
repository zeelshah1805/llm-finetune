"""Optional LLM-as-judge via a local Ollama server (PLAN.md §4.3).

Secondary, *relative* quality signal — never the headline (objective field-F1
stays the headline). Runs fully locally through Ollama (`http://localhost:11434`)
so there's no API key and no cost.

A judge model scores each predicted JSON 1–5 against the resume + gold JSON on a
documented rubric. If Ollama isn't running or `requests` is missing, judging is
skipped gracefully and the rest of the eval is unaffected.

Setup (once):
    # install Ollama from https://ollama.com, then:
    ollama pull llama3.1:8b      # or any chat model you have
    ollama serve                 # usually already running as a service
"""
from __future__ import annotations

import json
import re
from typing import Optional

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_MODEL = "llama3.1:8b"

RUBRIC = """\
You are grading how well a PREDICTED JSON parses a resume, given the resume text
and the GOLD (correct) JSON. Score 1-5 on overall correctness + faithfulness:

5 = matches gold on all fields; valid JSON; no invented info.
4 = essentially correct; one trivial slip (e.g., a minor formatting difference).
3 = mostly right but a real error (a wrong/missing field or one hallucinated value).
2 = several fields wrong or missing, or partially invalid structure.
1 = invalid JSON, wrong schema, or largely fabricated.

Respond with ONLY a JSON object: {"score": <1-5>, "reason": "<short>"}.
"""


def is_available(host: str = DEFAULT_HOST, timeout: float = 1.5) -> bool:
    """True if a local Ollama server is reachable."""
    try:
        import requests
        requests.get(f"{host}/api/tags", timeout=timeout)
        return True
    except Exception:  # noqa: BLE001  (no requests, connection refused, etc.)
        return False


def _extract_score(text: str) -> Optional[int]:
    # prefer a JSON object; fall back to the first 1-5 digit
    try:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            obj = json.loads(text[start:end + 1])
            s = int(obj.get("score"))
            if 1 <= s <= 5:
                return s
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    m = re.search(r"\b([1-5])\b", text)
    return int(m.group(1)) if m else None


def judge_one(resume: str, gold: dict, pred_raw: str,
              model: str = DEFAULT_MODEL, host: str = DEFAULT_HOST,
              timeout: float = 60.0) -> Optional[int]:
    """Return a 1–5 score for a single prediction, or None on failure."""
    import requests

    prompt = (
        f"{RUBRIC}\n\nRESUME:\n{resume}\n\n"
        f"GOLD JSON:\n{json.dumps(gold, ensure_ascii=False)}\n\n"
        f"PREDICTED (raw model output):\n{pred_raw}\n\nYour grade:"
    )
    try:
        resp = requests.post(
            f"{host}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False,
                  "options": {"temperature": 0.0}},
            timeout=timeout,
        )
        resp.raise_for_status()
        return _extract_score(resp.json().get("response", ""))
    except Exception:  # noqa: BLE001
        return None


def judge_predictions(rows: list[dict], model: str = DEFAULT_MODEL,
                      host: str = DEFAULT_HOST) -> Optional[dict]:
    """Score a list of {resume, target, raw} rows. Returns aggregate stats or
    None if the judge is unavailable / no rows could be scored.

    `resume` is optional in rows; if absent the judge grades pred vs. gold only.
    """
    if not is_available(host):
        print(f"(judge: no Ollama server at {host} — skipping LLM-as-judge)")
        return None

    scores: list[int] = []
    for i, r in enumerate(rows, 1):
        raw = r.get("raw")
        pred_raw = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)
        s = judge_one(r.get("resume", ""), r["target"], pred_raw, model=model, host=host)
        if s is not None:
            scores.append(s)
        if i % 10 == 0 or i == len(rows):
            print(f"  [judge] {i}/{len(rows)} scored")

    if not scores:
        print("(judge: no scores returned — skipping)")
        return None

    return {
        "judge_model": model,
        "judge_n": len(scores),
        "judge_mean": sum(scores) / len(scores),
        "judge_pct_5": sum(s == 5 for s in scores) / len(scores),
    }
