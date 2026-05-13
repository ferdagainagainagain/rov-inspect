gi# rov-inspect

A local pipeline that turns ROV underwater video and Deep Trekker telemetry into structured Italian inspection reports. A vision-language model classifies each candidate frame against a fixed Pydantic schema derived from 93 expert-curated reference frames; the output is a Markdown report with representative figures, depth, GPS, and timestamps. Everything runs on Apple Silicon — no API calls, no costs, no data leaving the device.

This project was built as a computer-vision course submission. The architecture decisions, debugging history, evaluation methodology, and known limitations are documented in [docs/PROJECT_LOG.md](docs/PROJECT_LOG.md). New collaborators should read that file first.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[local]"

python scripts/run_pipeline.py \
  --video "data/messina/VIDEO 1/videos/<filename>.mp4" \
  --log   "data/messina/VIDEO 1/data/combined_log.csv" \
  --out   out/video1_dev
```

Open `out/video1_dev/report.md`. First run downloads ~6 GB of model weights to `~/.cache/huggingface` and ~90 MB for the DINOv3 embedder. Subsequent runs are cached.

## Architecture overview

Three sequential stages:

1. **Sync + candidate extraction.** Align telemetry with video frames, score candidates by sharpness, motion, brightness, depth, and camera pitch. Keep the top 3 per 20-second bucket to enforce temporal coverage.
2. **Frame analysis.** Pass each candidate through Gemma 4 E4B (4-bit, via MLX) with a Pydantic schema describing the expected categorical output. Compute a DINOv3 ViT-S/16 embedding (via PyTorch + MPS) of each frame for downstream dedup.
3. **Segment + render.** Merge adjacent frames whose categorical *and* visual signatures both agree. Render the resulting segments into Italian Markdown via fixed templates (anti-hallucination — natural language is templated, only structured fields come from the VLM).

See [docs/PROJECT_LOG.md](docs/PROJECT_LOG.md) for why each stage is the way it is.

## Repository layout

```
src/rov_inspect/   Pipeline modules — schema, sync, candidates, enhance,
                   vlm_local, embed, segment, render, pipeline
scripts/           CLI entry points; run_pipeline.py is the main one
eval/              GT extraction, eval runner, F1/Jaccard comparator
docs/              PROJECT_LOG.md (decisions + design notes),
                   archive/ (superseded artifacts)
demo/              Committed example outputs — Video 3 report and one
                   SAM 3 verification figure
data/              Videos, telemetry, GPX — gitignored, provided externally
out/               Per-run reports and frames — gitignored
```

## Running the pipeline

Canonical invocation with the flags that affect output quality:

```bash
python scripts/run_pipeline.py \
  --video "data/messina/VIDEO 1/videos/<filename>.mp4" \
  --log   "data/messina/VIDEO 1/data/combined_log.csv" \
  --gpx   "data/messina/VIDEO 1/data/position/global/<filename>.gpx" \
  --out   out/video1_run \
  --model "mlx-community/gemma-4-e4b-it-4bit" \
  --bucket-seconds 20 --top-per-bucket 3 \
  --use-embeddings --embedding-threshold 0.50 \
  --enhance
```

The `--no-enhance`, `--no-use-embeddings`, and `--model` flags exist as deliberate ablation hooks — use them when comparing configurations. The default values above are the production-tuned ones; changing them will affect output quality and invalidate comparisons against the documented evaluation results.

## Running SAM 3 spatial verification

Optional post-processing layer that adds pixel-level outlines on top of the VLM's categorical claims. For each figure in `report.md`, the script reads the Italian `Descrizione` line, looks up which findings the VLM named (rocce / ciottoli / vegetazione / rete / tubo / condotta / rifiuto / relitto), runs SAM 3 with the corresponding English text prompt, and draws outline-only contours per class onto the frame. No fill — the original image stays fully visible inside each contour so you can judge SAM 3's localization without colour bleed.

### Setup

SAM 3 weights are gated on Hugging Face. Request access at <https://huggingface.co/facebook/sam3>, then authenticate locally:

```bash
hf auth login
```

The model (~1.5 GB) auto-downloads on first run to `~/.cache/huggingface`.

### Run

```bash
python scripts/run_sam3_postprocess.py --make-comparison
```

The script is currently hardcoded to `out/video3_enhanced` (a `--run-dir` argument is accepted but ignored). It writes to `out/video3_enhanced/sam3_verification/`:

- `figure_NN_overlay.png` — outline overlay
- `figure_NN_comparison.png` — side-by-side original | overlay (only with `--make-comparison`)
- `figure_NN_coverage.json` — matched VLM keywords, per-prompt coverage, per-class coverage
- `summary.json` — run-wide aggregate
- `sam_verification.md` — consolidated demo report linking the per-figure comparisons

### What to expect

SAM 3 runs ~2–4 s/frame on MPS. The selector only fires prompts the VLM hinted at, so most figures need only one or two prompts. Empirically on Messina data: `rocks` fires reliably on rocky frames (often 30–90% area), `vegetation` fires sparsely (0–3%), `cobbles` rarely grounds anything — see `docs/PROJECT_LOG.md` for the full empirical write-up and why fauna was deliberately dropped from the prompt set.

A committed example output lives at [demo/video3_enhanced/sam3_verification/figure_18_comparison.png](demo/video3_enhanced/sam3_verification/figure_18_comparison.png).

## Running the evaluation

The ground truth is extracted from a private docx; both the extractor and the 93-entry annotation JSON are tracked in git.

```bash
python eval/extract_gt.py --docx "data/messina/Rov Immagine.docx"
python eval/run_eval.py --config-name baseline_no_enhance --no-enhance --model "mlx-community/gemma-4-e4b-it-4bit"
python eval/run_eval.py --config-name baseline_enhanced --enhance --model "mlx-community/gemma-4-e4b-it-4bit"
python eval/compare.py eval/results/baseline_no_enhance.json eval/results/baseline_enhanced.json
```

`compare.py` prints per-field F1 / Jaccard, an equal-weight aggregate, and a per-entry disagreement breakdown. Always pass `--model` explicitly to avoid eval results drifting if a default changes.

Current baseline (Gemma 4 E4B, 4-bit, all 93 GT entries): aggregate F1 = 0.515 without enhancement, 0.535 with enhancement. Per-metric breakdown in [docs/PROJECT_LOG.md](docs/PROJECT_LOG.md) §3.4.

## Project log and decisions

[docs/PROJECT_LOG.md](docs/PROJECT_LOG.md) is the single most important document for a new collaborator. It explains why the pipeline looks the way it does, what we tried and rejected, and where the next interesting work lies (V-JEPA, Qwen3-VL comparison).

## Hardware requirements

Apple Silicon Mac, 24 GB unified memory recommended. The VLM runs via `mlx-vlm` (MLX backend); the DINOv3 embedder runs via PyTorch on the MPS backend. Both install transitively from the `[local]` extra — no manual GPU setup, no CUDA path supported.

Python 3.11+ required. Python 3.13 works but is the bleeding edge; if `pip install` fails on a dependency, see [docs/PROJECT_LOG.md](docs/PROJECT_LOG.md) §4.7 for known traps.

## License

MIT. See [LICENSE](LICENSE).
