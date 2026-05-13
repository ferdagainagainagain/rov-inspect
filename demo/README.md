# Demo outputs

Example artifacts produced by the pipeline. Live runs land under `out/`
(which is gitignored); the files here are committed snapshots so new
collaborators can see what the system produces without rerunning it.

## Contents

```
demo/video3_enhanced/
├── report.md                                       VLM-generated Italian inspection report (40 figures)
├── frames/                                         representative frames referenced by report.md
└── sam3_verification/
    ├── figure_18_comparison.png                   side-by-side: original | SAM 3 outline overlay
    ├── figure_18_overlay.png                       SAM 3 outline overlay only
    └── figure_18_coverage.json                     per-class coverage + matched VLM keywords
```

## How it was generated

The report came from:

```bash
python scripts/run_pipeline.py \
  --video "data/messina/VIDEO 3/videos/<filename>.mp4" \
  --log   "data/messina/VIDEO 3/data/combined_log.csv" \
  --gpx   "data/messina/VIDEO 3/data/position/global/<filename>.gpx" \
  --out   out/video3_enhanced \
  --model "mlx-community/gemma-4-e4b-it-4bit" \
  --enhance
```

The SAM 3 figure 18 came from running the spatial-verification post-processor on
that output:

```bash
python scripts/run_sam3_postprocess.py --make-comparison
```

Open `report.md` to see the full inspection report; open
`sam3_verification/figure_18_comparison.png` to see what SAM 3 outlines on
one of the frames the VLM flagged with discrete findings.
