"""Deterministically generate a synthetic resume -> JSON dataset.

Why synthetic: we get gold labels for free and full control over difficulty.
The generator renders the SAME structured record into several different free-text
resume layouts (classic, narrative, bullets, minimal, ...), which (a) gives the
fine-tune real signal to learn and (b) makes a base model visibly stumble on
formatting/consistency -- exactly the gap the eval framework measures.

Run from the repo root:

    python data/build_dataset.py --n 400 --seed 42

Outputs (frozen, version-controlled):
    data/train.jsonl   data/val.jsonl   data/test.jsonl

Each line is {"resume": <str>, "target": <gold JSON obj>}. The TEST split is the
integrity backbone (PLAN.md §2): it must never be seen during training or prompt
tuning.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

# Make the repo-root `schema` module importable when run as `python data/...`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from schema import TOP_LEVEL_KEYS, canonical_json  # noqa: E402

# ---------------------------------------------------------------------------
# Sampling pools
# ---------------------------------------------------------------------------
FIRST_NAMES = [
    "Aarav", "Diya", "Rohan", "Priya", "Arjun", "Sara", "Vikram", "Neha",
    "Karan", "Ananya", "James", "Emily", "Michael", "Olivia", "Daniel",
    "Sophia", "Wei", "Mei", "Chen", "Yuki", "Carlos", "Sofia", "Omar",
    "Layla", "Liam", "Emma", "Noah", "Ava", "Ethan", "Isabella",
]
LAST_NAMES = [
    "Sharma", "Patel", "Reddy", "Iyer", "Gupta", "Singh", "Nair", "Mehta",
    "Smith", "Johnson", "Williams", "Brown", "Garcia", "Martinez", "Lee",
    "Kim", "Wang", "Zhang", "Khan", "Ali", "Anderson", "Thomas", "Chen",
]
EMAIL_DOMAINS = ["gmail.com", "outlook.com", "yahoo.com", "proton.me", "hotmail.com"]

SKILLS = {
    "lang": ["Python", "Java", "C++", "Go", "Rust", "JavaScript", "TypeScript", "SQL", "Scala", "R"],
    "ml": ["PyTorch", "TensorFlow", "scikit-learn", "Hugging Face", "LangChain", "XGBoost", "Keras"],
    "data": ["Spark", "Airflow", "dbt", "Snowflake", "Kafka", "Pandas", "BigQuery"],
    "cloud": ["AWS", "GCP", "Azure", "Docker", "Kubernetes", "Terraform"],
    "web": ["React", "Node.js", "Django", "FastAPI", "Flask", "GraphQL", "Redis", "PostgreSQL"],
}

COMPANIES = [
    "Infosys", "TCS", "Wipro", "Flipkart", "Zomato", "Razorpay", "Swiggy",
    "Google", "Microsoft", "Amazon", "Stripe", "Datadog", "Atlassian",
    "Acme Corp", "Nimbus Labs", "BlueOcean Tech", "Quantix", "Helix Systems",
]
TITLES = [
    "Software Engineer", "Senior Software Engineer", "Data Scientist",
    "Machine Learning Engineer", "Backend Developer", "Data Engineer",
    "Full Stack Developer", "DevOps Engineer", "Research Engineer",
    "Analytics Lead", "Platform Engineer",
]
UNIVERSITIES = [
    "IIT Bombay", "IIT Delhi", "NIT Trichy", "BITS Pilani", "VIT Vellore",
    "Stanford University", "MIT", "UC Berkeley", "University of Toronto",
    "Georgia Tech", "Delhi University", "Anna University",
]
DEGREES = [
    "B.Tech in Computer Science", "B.E. in Information Technology",
    "B.Sc in Computer Science", "M.Tech in Data Science",
    "M.S. in Computer Science", "MBA", "B.Tech in Electronics",
]

CITIES = ["Bengaluru", "Mumbai", "Hyderabad", "Pune", "Delhi", "Chennai",
          "Remote", "London", "Toronto", "Singapore"]

# ---------------------------------------------------------------------------
# Record generation (the gold structured truth)
# ---------------------------------------------------------------------------


def _phone(rng: random.Random) -> str:
    return f"+91-{rng.randint(70000, 99999)}-{rng.randint(10000, 99999)}"


def _make_skills(rng: random.Random) -> list[str]:
    cats = rng.sample(list(SKILLS), k=rng.randint(2, 4))
    out: list[str] = []
    for c in cats:
        out += rng.sample(SKILLS[c], k=rng.randint(1, 3))
    # dedupe while preserving order
    seen, deduped = set(), []
    for s in out:
        if s not in seen:
            seen.add(s)
            deduped.append(s)
    return deduped


def generate_record(rng: random.Random) -> dict:
    """Produce one gold JSON record matching schema.py's contract."""
    name = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
    has_email = rng.random() > 0.12
    has_phone = rng.random() > 0.30

    n_jobs = rng.randint(1, 4)
    work = []
    for _ in range(n_jobs):
        work.append({
            "title": rng.choice(TITLES),
            "company": rng.choice(COMPANIES),
            "years": rng.choice([1, 1.5, 2, 2.5, 3, 4, 5]),
        })
    total_years = round(sum(w["years"] for w in work), 1)
    # store as int when whole, mirroring how humans write it
    if total_years == int(total_years):
        total_years = int(total_years)

    n_edu = rng.randint(1, 2)
    edu = []
    for _ in range(n_edu):
        has_year = rng.random() > 0.15
        edu.append({
            "degree": rng.choice(DEGREES),
            "institution": rng.choice(UNIVERSITIES),
            "year": rng.randint(2008, 2023) if has_year else None,
        })

    local = name.lower().split()[0]
    email = f"{local}.{rng.randint(1, 999)}@{rng.choice(EMAIL_DOMAINS)}" if has_email else None

    return {
        "name": name,
        "email": email,
        "phone": _phone(rng) if has_phone else None,
        "total_years_experience": total_years,
        "skills": _make_skills(rng),
        "education": edu,
        "work_experience": work,
    }


# ---------------------------------------------------------------------------
# Text rendering -- several layouts so the input distribution is varied
# ---------------------------------------------------------------------------


def _render_classic(rec: dict, rng: random.Random) -> str:
    lines = [rec["name"].upper(), rng.choice(CITIES)]
    contact = []
    if rec["email"]:
        contact.append(rec["email"])
    if rec["phone"]:
        contact.append(rec["phone"])
    if contact:
        lines.append(" | ".join(contact))
    lines.append("")
    lines.append("SUMMARY")
    lines.append(f"{rec['total_years_experience']} years of professional experience.")
    lines.append("")
    lines.append("EXPERIENCE")
    for w in rec["work_experience"]:
        lines.append(f"- {w['title']}, {w['company']} ({w['years']} yrs)")
    lines.append("")
    lines.append("EDUCATION")
    for e in rec["education"]:
        yr = f", {e['year']}" if e["year"] is not None else ""
        lines.append(f"- {e['degree']}, {e['institution']}{yr}")
    lines.append("")
    lines.append("SKILLS: " + ", ".join(rec["skills"]))
    return "\n".join(lines)


def _render_narrative(rec: dict, rng: random.Random) -> str:
    parts = [f"{rec['name']} is a professional with {rec['total_years_experience']} "
             f"years of experience based in {rng.choice(CITIES)}."]
    jobs = "; ".join(f"{w['title']} at {w['company']} for {w['years']} years"
                     for w in rec["work_experience"])
    parts.append(f"Roles held: {jobs}.")
    edus = "; ".join(
        f"{e['degree']} from {e['institution']}" + (f" ({e['year']})" if e["year"] else "")
        for e in rec["education"]
    )
    parts.append(f"Education: {edus}.")
    parts.append(f"Key skills include {', '.join(rec['skills'])}.")
    contact = []
    if rec["email"]:
        contact.append(f"reachable at {rec['email']}")
    if rec["phone"]:
        contact.append(f"phone {rec['phone']}")
    if contact:
        parts.append("Contact: " + " and ".join(contact) + ".")
    return " ".join(parts)


def _render_bullets(rec: dict, rng: random.Random) -> str:
    head = rec["name"]
    if rec["email"]:
        head += f"  •  {rec['email']}"
    if rec["phone"]:
        head += f"  •  {rec['phone']}"
    lines = [head, ""]
    lines.append("Work History:")
    for w in rec["work_experience"]:
        lines.append(f"  * {w['company']} — {w['title']} — {w['years']} year(s)")
    lines.append("")
    lines.append("Academics:")
    for e in rec["education"]:
        yr = f" [{e['year']}]" if e["year"] is not None else ""
        lines.append(f"  * {e['institution']}: {e['degree']}{yr}")
    lines.append("")
    lines.append(f"Total Experience: {rec['total_years_experience']} years")
    lines.append("Tech: " + " / ".join(rec["skills"]))
    return "\n".join(lines)


def _render_minimal(rec: dict, rng: random.Random) -> str:
    # Terse, low-structure, hardest case for a base model.
    bits = [rec["name"]]
    if rec["email"]:
        bits.append(rec["email"])
    bits.append(f"{rec['total_years_experience']}y exp")
    bits.append("Skills: " + ",".join(rec["skills"]))
    jobs = " ".join(f"{w['title']}@{w['company']}({w['years']}y)" for w in rec["work_experience"])
    bits.append(jobs)
    edu = " ".join(f"{e['degree']}-{e['institution']}" + (f"-{e['year']}" if e["year"] else "")
                   for e in rec["education"])
    bits.append(edu)
    return " | ".join(bits)


RENDERERS = [_render_classic, _render_narrative, _render_bullets, _render_minimal]


def render_resume(rec: dict, rng: random.Random) -> str:
    return rng.choice(RENDERERS)(rec, rng)


# ---------------------------------------------------------------------------
# Build + split + write
# ---------------------------------------------------------------------------


def build(n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    examples = []
    for _ in range(n):
        rec = generate_record(rng)
        # validate key set so generation can never drift from the schema
        assert set(rec) == set(TOP_LEVEL_KEYS), "record keys drifted from schema"
        examples.append({"resume": render_resume(rec, rng), "target": rec})
    return examples


def split(examples: list[dict], seed: int) -> dict[str, list[dict]]:
    rng = random.Random(seed + 1)  # distinct stream from generation
    idx = list(range(len(examples)))
    rng.shuffle(idx)
    n = len(idx)
    n_train, n_val = int(0.8 * n), int(0.1 * n)
    parts = {
        "train": idx[:n_train],
        "val": idx[n_train:n_train + n_val],
        "test": idx[n_train + n_val:],
    }
    return {k: [examples[i] for i in v] for k, v in parts.items()}


def write_jsonl(path: str, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the synthetic resume->JSON dataset.")
    ap.add_argument("--n", type=int, default=400, help="total examples to generate")
    ap.add_argument("--seed", type=int, default=42, help="deterministic seed")
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)),
                    help="output directory (default: data/)")
    args = ap.parse_args()

    examples = build(args.n, args.seed)
    splits = split(examples, args.seed)

    for name, rows in splits.items():
        path = os.path.join(args.out, f"{name}.jsonl")
        write_jsonl(path, rows)
        print(f"wrote {len(rows):4d} -> {path}")

    # tiny sanity peek
    sample = splits["train"][0]
    print("\nExample resume:\n" + "-" * 60)
    print(sample["resume"][:400])
    print("-" * 60 + "\nTarget JSON:")
    print(canonical_json(sample["target"]))


if __name__ == "__main__":
    main()
