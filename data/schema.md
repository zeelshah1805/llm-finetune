# Resume → JSON schema

The model is fine-tuned to convert free-text resumes into one JSON object with
these **fixed top-level keys** (no extras, no missing keys):

| Key | Type | Notes |
|---|---|---|
| `name` | string | Candidate full name. |
| `email` | string \| null | `null` if not present in the resume. |
| `phone` | string \| null | `null` if not present. Digits/format preserved as written. |
| `total_years_experience` | number | Total professional experience in years (may be fractional). |
| `skills` | array of string | Deduplicated technical/professional skills. |
| `education` | array of object | Each: `{ "degree", "institution", "year" }`. |
| `work_experience` | array of object | Each: `{ "title", "company", "years" }`. |

### Nested objects

`education[]` items:

| Key | Type |
|---|---|
| `degree` | string |
| `institution` | string |
| `year` | number \| null (graduation year) |

`work_experience[]` items:

| Key | Type |
|---|---|
| `title` | string |
| `company` | string |
| `years` | number (years in that role) |

### Rules

- Output is a **single JSON object only** — no markdown fences, no prose.
- Missing `email`, `phone`, or graduation `year` → `null`.
- Numbers are JSON numbers, never strings (`5`, not `"5"`).
- `skills` is deduplicated and order-insensitive (scored as a set).

### How this maps to evaluation

| Field | Scoring (see `eval/metrics.py`) |
|---|---|
| `name`, `email`, `phone` | normalized exact match |
| `total_years_experience` | numeric match (tolerance ±0.5) |
| `skills` | set precision / recall / **F1** |
| `education`, `work_experience` | list-of-objects F1 (greedy best-match on items) |

The aggregate **field-level F1** across all keys is the headline accuracy
metric. JSON-validity rate and hallucinated/extra-field rate are computed
separately as the format-failure metrics.
