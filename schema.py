"""Single source of truth for the resume-parsing task.

Both the data pipeline (data/build_dataset.py) and the eval framework
(eval/*) import from here so the schema, the prompt, and the chat
formatting never drift apart. This is the integrity backbone referenced
in PLAN.md §2.
"""
from __future__ import annotations

import json
from typing import Any

# ---------------------------------------------------------------------------
# The JSON schema. Fixed keys, documented in data/schema.md. Ambiguity here
# destroys eval reliability, so keep it small and rigid.
# ---------------------------------------------------------------------------
TOP_LEVEL_KEYS: list[str] = [
    "name",
    "email",
    "phone",
    "total_years_experience",
    "skills",
    "education",
    "work_experience",
]

EDUCATION_KEYS: list[str] = ["degree", "institution", "year"]
WORK_KEYS: list[str] = ["title", "company", "years"]

# A compact, machine-readable contract we paste into the system prompt so the
# model knows exactly which keys to emit. Kept terse to save tokens.
SCHEMA_HINT: str = json.dumps(
    {
        "name": "string",
        "email": "string or null",
        "phone": "string or null",
        "total_years_experience": "number",
        "skills": ["string"],
        "education": [{"degree": "string", "institution": "string", "year": "number or null"}],
        "work_experience": [{"title": "string", "company": "string", "years": "number"}],
    },
    indent=2,
)

SYSTEM_PROMPT: str = (
    "You are a precise resume parser. Read the resume text and return ONLY a "
    "single JSON object that matches this schema exactly, with no extra keys, "
    "no markdown, and no commentary:\n"
    f"{SCHEMA_HINT}\n"
    "Rules: use null for missing email/phone/year. skills is a deduplicated "
    "list of strings. Numbers must be numbers, not strings. Output JSON only."
)


def build_user_prompt(resume_text: str) -> str:
    """The user turn: the raw resume the model must parse."""
    return f"Resume:\n{resume_text.strip()}\n\nReturn the JSON object."


def build_messages(resume_text: str) -> list[dict[str, str]]:
    """Chat-format messages (system + user). Pass through
    tokenizer.apply_chat_template to get the model-specific prompt string."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(resume_text)},
    ]


def canonical_json(obj: dict[str, Any]) -> str:
    """Stable serialization of a gold/predicted record (used as the training
    target completion). Sorted keys + compact-ish indent for readability."""
    return json.dumps(obj, ensure_ascii=False, indent=2)
