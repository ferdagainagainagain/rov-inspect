# Project context for Claude Code

## What this project is

End-to-end pipeline that turns ROV underwater video + Deep Trekker
telemetry into an Italian inspection report.

Domain: Sicilian coastal seabed surveys (port and seabed engineering).
The user is a project engineer at SIGMA Ingegneria; this is built as
a Computer Vision course project that should also be useful in the
firm's actual reporting workflow.

## Hard constraint: NO paid APIs

This is a university project with a fairness rule: nothing that costs
money may be in the working pipeline (other teams must not be
disadvantaged). The default and only required path is fully local on
Apple Silicon.

- **Default backend: `vlm_local.py`** (mlx-vlm + Qwen2.5-VL-7B-4bit)
- `vlm_api.py` exists only for *ablation comparisons in the writeup*.
  Never make it the default, never run it as part of normal dev,
  never assume an API key is available.

If a task seems to require external services (translation, image
search, etc.), **stop and ask** rather than reaching for an API.

## Target hardware

M4 with 24 GB unified memory. Qwen2.5-VL-7B-4bit (~6 GB) is the
default; the 3B variant is available via `--fast`. Don't propose
larger quantizations or models without checking memory budgets.

## Ground truth

`Rov_Immagine.docx` (kept outside this repo, ~31 MB): 93 captioned
figures across 11 videos. Italian, controlled vocabulary covering
seabed type, granulometry, algal coverage and taxa, anthropogenic
objects, and fauna.

The schema in `src/rov_inspect/schema.py` mirrors this vocabulary
**exactly**. When extending, add new enum values (matching GT spelling
and case) rather than freeform strings. New value, new enum entry —
this is the single most important convention.

## Code conventions

- Python 3.11+, type hints, `from __future__ import annotations`.
- Dataclasses for transient containers; Pydantic only for VLM I/O validation.
- Italian preserved in user-facing strings (captions, prompts) and enum
  values; English for code-internal names.
- No global state. Functions take explicit paths and return explicit objects.
- OpenCV BGR convention; convert at boundaries if needed.

## What NOT to do without asking

- Do **not** check video files or `combined_log.csv` into git.
- Do **not** change the default backend from local to api.
- Do **not** generate freeform Italian prose in `render.py`. The
  template-based approach is intentional anti-hallucination.
- Do **not** add new categorical values that aren't attested in the GT
  without flagging it.
- Do **not** import `anthropic` outside `vlm_api.py`.
- Do **not** change the DINOv3 model size without confirming — the
  transformers/PyTorch-MPS path has fixed model names and memory
  footprints that matter.

## Common tasks

- "Run on a different video": pass `--video` and `--log`. No code change.
- "Captions read awkwardly": tweak `_*_LABEL` mappings in `render.py`,
  not the schema.
- "VLM keeps misclassifying X": refine the prompt in
  `src/rov_inspect/vlm_common.py`. Add a guideline bullet, not a
  hardcoded rule.
- "Add a new species/object": new enum value in `schema.py`, new label
  entry in `render.py`.

## Open questions to confirm with the user before changing

- Telemetry column names depend on Deep Trekker firmware. Don't guess —
  run `scripts/explore_logs.py` first.
- Whether `position/global` (surface GPS) or `position/local` (IMU/DVL)
  is the right positional source.
