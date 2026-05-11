# rov-inspect

AI-assisted ROV video inspection and reporting — **fully local, no API costs**.

**Pipeline:** ROV video + Deep Trekker telemetry → structured frame analyses
(Italian, GT-aligned vocabulary) → Markdown report with representative
frames, depth, GPS coordinates, and timestamps.

**Design constraint:** This is a university project. The default and only
required path is fully local inference on Apple Silicon. No external APIs,
no costs, no data leaving the device.

## Status

**v0.2** — local-first, single-video demo. Recommended first target:
`VIDEO 1` from the Messina dataset, first ~2 minutes of the SD video,
to validate the schema before scaling to longer footage.

## Stack

| Component  | Choice                                         | Why                                                  |
|------------|------------------------------------------------|------------------------------------------------------|
| VLM        | Qwen2.5-VL-7B-Instruct-4bit (via mlx-vlm)      | Strong VLM, multilingual, ~6 GB, 3-8 s/frame on M4   |
| Fast VLM   | Qwen2.5-VL-3B-Instruct-4bit                    | Quicker iteration, lower quality                     |
| Schema     | Pydantic enums built from the 93-figure GT vocab | Prevents freeform hallucination                    |
| Rendering  | Templated Italian from structured fields       | Stylistic fidelity to GT, no LLM in the prose path   |

## Setup (M-series Mac)

Requires Python 3.11+ and Apple Silicon. With [`uv`](https://docs.astral.sh/uv/) (recommended):

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[local]"
```

That's it. No API keys. The Qwen weights (~5 GB) will auto-download
to `~/.cache/huggingface` on the first VLM call.

## First run (do this in order)

### 1) Inspect your telemetry log

```bash
python scripts/explore_logs.py "/path/to/VIDEO 1/data/combined_log.csv"
```

Look at the printed column list and update `COLUMN_MAP_DEFAULT` in
`src/rov_inspect/sync.py`. Minimum: a `t_sec` column (or ISO timestamp
the loader can convert). Useful extras: `depth_m`, `lat`, `lon`,
`heading_deg`.

### 2) Smoke-test the local VLM on one frame

Pick any frame from a video (e.g. extract with `ffmpeg -ss 30 -i video.mp4 -vframes 1 test.jpg`):

```bash
python scripts/test_local_vlm.py test.jpg
```

This downloads the model (~5 GB, one-time) and runs inference on one
image. You'll see the structured JSON output and the rendered Italian
caption. If this works, the pipeline will work.

### 3) Run on a 2-minute slice of VIDEO 1 (SD)

```bash
python scripts/run_pipeline.py \
  --video "/path/to/VIDEO 1/videos/..._SD.mp4" \
  --log   "/path/to/VIDEO 1/data/combined_log.csv" \
  --out   out/video1_dev \
  --start 0 --end 120
```

Outputs:
- `out/video1_dev/report.md` — the inspection report
- `out/video1_dev/frames/*.jpg` — selected representative frames

### 4) Compare against the ground truth

Open `Rov_Immagine.docx`, find the "video 1" section, and compare the
captions against `report.md`. Note where they differ — those are
signals for prompt or schema refinement.

## Performance on M4 (24 GB)

| Setting                | Calls (2 min @ 1 fps, top 6/min) | Wall time     |
|------------------------|----------------------------------|---------------|
| `--fast` (3B)          | ~12                              | ~2-4 min      |
| default (7B, 4-bit)    | ~12                              | ~5-10 min     |

First run adds ~1-2 minutes for model download. Full 30-minute video
at default: 30-60 minutes of inference — run during a coffee break or
overnight.

## Architecture

```
sync.py        log + video alignment, per-frame interpolation
candidates.py  frame extraction + quality scoring (sharpness, speed, brightness)
vlm_common.py  shared prompts and response parsing
vlm_local.py   mlx-vlm + Qwen2.5-VL — DEFAULT BACKEND
vlm_api.py     Anthropic API — kept for reference only, not used
segment.py     merge adjacent same-content frames; pick representative
render.py      structured analysis → Italian caption + Markdown report
pipeline.py    end-to-end orchestration
```

The schema (`schema.py`) is built from the GT vocabulary in
`Rov_Immagine.docx` (93 figures across 11 videos). When you discover
a missing term, **add it to the schema** rather than letting the VLM
emit freeform strings.

## Notes on the API backend

`vlm_api.py` exists for one reason: **comparative ablation in the
writeup**. If you want to show "our local Qwen-7B achieves X% of the
quality of Claude Sonnet at zero cost", you can run a single short
slice with `--backend api`. This is not needed to run the project.
The local backend is the deliverable.

## Roadmap

- [ ] DINOv2 / CLIP embeddings (local) in `segment.py` for content-based dedup
- [ ] V-JEPA short-clip embeddings for change detection
- [ ] `eval/compare_to_gt.py` — quantitative evaluation against the GT
- [ ] `.docx` output mirroring the GT layout
- [ ] Spatial map (folium) of findings using GPS coords
