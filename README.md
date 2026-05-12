# rov-inspect

Local pipeline that turns ROV underwater video and Deep Trekker
telemetry into a structured Italian inspection report. A vision-language
model classifies each candidate frame against a fixed schema derived
from a 93-figure expert ground truth; the result is a Markdown report
with representative frames, depth, GPS, and timestamps. Everything runs
on Apple Silicon — no API calls, no costs, no data leaving the device.

## Quick start

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[local]"
python scripts/run_pipeline.py \
  --video "data/messina/VIDEO 1/videos/..._SD.mp4" \
  --log   "data/messina/VIDEO 1/data/combined_log.csv" \
  --out   out/video1_dev --start 0 --end 120
```

Open `out/video1_dev/report.md`. First run downloads ~6 GB of model
weights to `~/.cache/huggingface`.

## Architecture overview

The pipeline has three stages: **(1) sync + candidate extraction**
(align telemetry with video, sample sharp frames with downward camera
tilt and reasonable depth), **(2) VLM analysis + embedding** (run
Qwen2.5-VL on each candidate to produce a Pydantic-validated
`FrameAnalysis`, and DINOv3 to embed it for content-aware dedup), and
**(3) segment + render** (merge adjacent frames whose categorical and
visual signatures agree, then emit Italian prose from a template).
Background on why each stage exists lives in
[docs/PROJECT_LOG.md](docs/PROJECT_LOG.md).

## Repository layout

```
src/rov_inspect/   pipeline modules — sync, candidates, vlm_*, embed, segment, render
scripts/           CLI entry points (run_pipeline.py is the main one)
eval/              ground-truth extractor, VLM runner, F1 comparator
data/              videos, logs, GPX — gitignored; provided externally
out/               per-run reports + extracted frames — gitignored
docs/              project log, archived artifacts
```

## Running the pipeline

Canonical invocation, with the flags that matter for quality:

```bash
python scripts/run_pipeline.py \
  --video "data/messina/VIDEO 1/videos/..._SD.mp4" \
  --log   "data/messina/VIDEO 1/data/combined_log.csv" \
  --gpx   "data/messina/VIDEO 1/data/position/global/global_position_log_*.gpx" \
  --out   out/video1_run \
  --model mlx-community/Qwen2.5-VL-7B-Instruct-4bit \
  --bucket-seconds 20 --top-per-bucket 3 \
  --use-embeddings --embedding-threshold 0.92 \
  --enhance
```

`--no-enhance`, `--no-use-embeddings`, and the `--model` override are
deliberate ablation hooks — use them when comparing configurations.
`--fast` selects the 3B variant for quicker iteration.

## Running the evaluation

The ground truth is a private docx; extraction is deterministic and the
two annotation JSONs are tracked.

```bash
python eval/extract_gt.py --docx ../Rov_Immagine.docx
# fill in eval/gt_annotations_subset.json by hand once
python eval/run_eval.py --config-name baseline --no-enhance
python eval/run_eval.py --config-name enhanced --enhance
python eval/compare.py eval/results/baseline.json eval/results/enhanced.json
```

`compare.py` prints per-field F1 / Jaccard, an equal-weight aggregate,
and a per-entry disagreement breakdown.

## Project log and decisions

Non-obvious design choices and open questions live in
[docs/PROJECT_LOG.md](docs/PROJECT_LOG.md).

## Hardware requirements

Apple Silicon Mac, 24 GB unified memory recommended. The VLM uses MLX
(via `mlx-vlm`); the DINOv3 embedder uses PyTorch on the MPS backend.
Both load transitively from the `[local]` extra — no manual GPU setup,
no CUDA path supported.

## License

MIT. See [LICENSE](LICENSE).
