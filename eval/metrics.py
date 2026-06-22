"""Metrics for the resume -> JSON task (PLAN.md §4).

Pure, dependency-free functions so they're trivially unit-testable and run
anywhere (no GPU, no torch). Everything is computed on the SAME frozen test set
for both the base and the fine-tuned model.

Three families of metric:
  1. Task accuracy   -> field-level F1 (the headline), plus per-field breakdown.
  2. Format failure  -> JSON-validity rate, schema-conformance, hallucinated
                        (extra) field rate. Fine-tuning should crush these.
  3. Efficiency      -> latency / output length (collected in run_eval, not here).
"""
from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any, Optional

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from schema import TOP_LEVEL_KEYS  # noqa: E402

YEAR_TOLERANCE = 0.5  # ±0.5 yr counts as a match for experience numbers

# ---------------------------------------------------------------------------
# 1. Extracting JSON from raw model output
# ---------------------------------------------------------------------------

_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def extract_json(text: str) -> Optional[dict]:
    """Best-effort parse of a model's raw text into a dict.

    Handles the common base-model failure modes: ```json fences, leading prose
    before the object, trailing commentary after it. Returns None if no valid
    JSON object can be recovered (this is itself a measured failure).
    """
    if not text:
        return None

    # 1) fenced block, if present
    m = _FENCE.search(text)
    candidates = [m.group(1)] if m else []

    # 2) the substring from the first '{' to the last '}'
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(text[start:end + 1])

    # 3) the whole thing, as a last resort
    candidates.append(text.strip())

    for cand in candidates:
        try:
            obj = json.loads(cand)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(obj, dict):
            return obj
    return None


# ---------------------------------------------------------------------------
# 2. Format-failure checks
# ---------------------------------------------------------------------------


def schema_report(obj: Optional[dict]) -> dict:
    """Conformance of a parsed object to the fixed key set."""
    if obj is None:
        return {"valid_json": False, "schema_ok": False,
                "missing_keys": list(TOP_LEVEL_KEYS), "extra_keys": []}
    keys = set(obj)
    expected = set(TOP_LEVEL_KEYS)
    missing = sorted(expected - keys)
    extra = sorted(keys - expected)  # hallucinated / invented fields
    return {
        "valid_json": True,
        "schema_ok": not missing and not extra,
        "missing_keys": missing,
        "extra_keys": extra,
    }


# ---------------------------------------------------------------------------
# 3. Field-level scoring via atom matching
# ---------------------------------------------------------------------------


def _norm(s: Any) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower()) if s is not None else "∅null"


def _phone_norm(s: Any) -> str:
    return re.sub(r"\D", "", str(s)) if s is not None else "∅null"


def _num(x: Any) -> Optional[float]:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def atoms(record: dict) -> Counter:
    """Decompose a record into a multiset of comparable atoms.

    Field-level F1 is then micro-averaged: TP = atoms common to gold & pred.
    Years are bucketed to the tolerance grid so ±0.5 differences still match.
    """
    c: Counter = Counter()
    c[("name", _norm(record.get("name")))] += 1
    c[("email", _norm(record.get("email")))] += 1
    c[("phone", _phone_norm(record.get("phone")))] += 1

    ty = _num(record.get("total_years_experience"))
    ty_key = round(ty / YEAR_TOLERANCE) if ty is not None else "∅null"
    c[("total_years", ty_key)] += 1

    for s in record.get("skills") or []:
        c[("skill", _norm(s))] += 1

    for e in record.get("education") or []:
        if isinstance(e, dict):
            c[("edu", _norm(e.get("degree")), _norm(e.get("institution")), _norm(e.get("year")))] += 1

    for w in record.get("work_experience") or []:
        if isinstance(w, dict):
            yr = _num(w.get("years"))
            yk = round(yr / YEAR_TOLERANCE) if yr is not None else "∅null"
            c[("work", _norm(w.get("title")), _norm(w.get("company")), yk)] += 1
    return c


def _prf(tp: int, n_pred: int, n_gold: int) -> tuple[float, float, float]:
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def score_example(pred_obj: Optional[dict], gold: dict) -> dict:
    """All per-example signals. Counts are returned so the dataset-level
    aggregate can micro-average correctly."""
    rep = schema_report(pred_obj)
    gold_atoms = atoms(gold)

    if pred_obj is None:
        pred_atoms: Counter = Counter()
    else:
        pred_atoms = atoms(pred_obj)

    overlap = pred_atoms & gold_atoms  # multiset intersection
    tp = sum(overlap.values())
    n_pred = sum(pred_atoms.values())
    n_gold = sum(gold_atoms.values())
    p, r, f1 = _prf(tp, n_pred, n_gold)

    # per-field-group breakdown for the detailed table
    def group_counts(prefix: str) -> tuple[int, int, int]:
        gp = Counter({k: v for k, v in pred_atoms.items() if k[0] == prefix})
        gg = Counter({k: v for k, v in gold_atoms.items() if k[0] == prefix})
        ov = sum((gp & gg).values())
        return ov, sum(gp.values()), sum(gg.values())

    return {
        "valid_json": rep["valid_json"],
        "schema_ok": rep["schema_ok"],
        "n_extra_keys": len(rep["extra_keys"]),
        "tp": tp, "n_pred": n_pred, "n_gold": n_gold,
        "precision": p, "recall": r, "field_f1": f1,
        "groups": {g: group_counts(g) for g in
                   ("name", "email", "phone", "total_years", "skill", "edu", "work")},
    }


# ---------------------------------------------------------------------------
# Dataset-level aggregation
# ---------------------------------------------------------------------------

_GROUP_LABELS = {
    "name": "name", "email": "email", "phone": "phone",
    "total_years": "total_years_experience", "skill": "skills",
    "edu": "education", "work": "work_experience",
}


def aggregate(per_example: list[dict], latencies: Optional[list[float]] = None,
              out_tokens: Optional[list[int]] = None) -> dict:
    """Roll per-example results up into the headline numbers (micro-averaged)."""
    n = len(per_example)
    if n == 0:
        return {}

    tp = sum(e["tp"] for e in per_example)
    n_pred = sum(e["n_pred"] for e in per_example)
    n_gold = sum(e["n_gold"] for e in per_example)
    p, r, f1 = _prf(tp, n_pred, n_gold)

    group_f1 = {}
    for g, label in _GROUP_LABELS.items():
        gtp = sum(e["groups"][g][0] for e in per_example)
        gpr = sum(e["groups"][g][1] for e in per_example)
        ggo = sum(e["groups"][g][2] for e in per_example)
        group_f1[label] = _prf(gtp, gpr, ggo)[2]

    summary = {
        "n": n,
        "json_validity": sum(e["valid_json"] for e in per_example) / n,
        "schema_conformance": sum(e["schema_ok"] for e in per_example) / n,
        # format-failure: invalid JSON OR wrong key set OR any extra (hallucinated) key
        "format_failure_rate": sum(
            not e["valid_json"] or not e["schema_ok"] or e["n_extra_keys"] > 0
            for e in per_example) / n,
        "hallucinated_field_rate": sum(e["n_extra_keys"] > 0 for e in per_example) / n,
        "field_precision": p,
        "field_recall": r,
        "field_f1": f1,
        "per_field_f1": group_f1,
    }
    if latencies:
        summary["avg_latency_s"] = sum(latencies) / len(latencies)
    if out_tokens:
        summary["avg_output_tokens"] = sum(out_tokens) / len(out_tokens)
    return summary


# ---------------------------------------------------------------------------
# Self-test: scoring is correct when run directly (no model needed)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    gold = {
        "name": "Jane Doe", "email": "jane@x.com", "phone": "+1 555 1234",
        "total_years_experience": 5,
        "skills": ["Python", "SQL"],
        "education": [{"degree": "BS CS", "institution": "MIT", "year": 2015}],
        "work_experience": [{"title": "Engineer", "company": "Acme", "years": 5}],
    }

    # perfect prediction -> F1 == 1.0
    perfect = score_example(json.loads(json.dumps(gold)), gold)
    assert abs(perfect["field_f1"] - 1.0) < 1e-9, perfect
    assert perfect["schema_ok"]

    # invalid JSON -> caught
    assert extract_json("sorry, here is the data: not json") is None

    # fenced + extra key + one wrong skill
    raw = '```json\n{"name":"Jane Doe","email":"jane@x.com","phone":"5551234",' \
          '"total_years_experience":5,"skills":["Python","Java"],' \
          '"education":[{"degree":"BS CS","institution":"MIT","year":2015}],' \
          '"work_experience":[{"title":"Engineer","company":"Acme","years":5}],' \
          '"summary":"hallucinated"}\n```'
    obj = extract_json(raw)
    sc = score_example(obj, gold)
    assert sc["valid_json"] and not sc["schema_ok"] and sc["n_extra_keys"] == 1
    assert 0.0 < sc["field_f1"] < 1.0  # one skill wrong

    agg = aggregate([perfect, sc], latencies=[0.5, 0.7], out_tokens=[80, 95])
    print("self-test OK")
    print(json.dumps(agg, indent=2))
