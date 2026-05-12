# Project Log: AI-Assisted ROV Underwater Inspection

**rov-inspect** — local-only pipeline that ingests ROV video + telemetry and produces structured Italian environmental inspection reports.

Repository: https://github.com/ferdagainagainagain/rov-inspect

---

## 1. What this project does

The Italian engineering firm SIGMA Ingegneria conducts ROV-based seabed surveys for port and coastal infrastructure projects. The current human workflow is: an expert biologist watches several hours of ROV footage, selects representative frames, and writes structured Italian captions describing substrate type, algal cover, fauna, and anthropogenic items. The Messina survey dataset used here consists of 11 videos and 93 expert-annotated reference frames.

This project automates that workflow as far as a zero-shot pipeline can — pulling representative frames, geo-anchoring them via GPS/depth telemetry, and emitting Pydantic-validated captions that mirror the expert vocabulary. The output is a Markdown inspection report, one figure per representative frame, with timestamps, coordinates, depth, and structured Italian description.

The project runs entirely locally on a 24 GB M4 MacBook with no paid API calls. 
---

## 2. Architecture

Three sequential stages, each addressing one distinct CV/ML problem:

**Stage 1 — Frame selection (classical CV + telemetry).**
The pipeline samples one frame per second from the video, scores each candidate using sharpness (Laplacian variance), motion (low ROV speed = better), brightness, depth, and camera pitch from the IMU, then keeps the top 3 candidates per 20-second bucket. The bucket-based sampling is one of the more important fixes documented below.

**Stage 2 — Frame analysis (VLM + structured output).**
Each selected frame is passed to a locally-hosted vision-language model (Gemma 4 E4B, 4-bit quantized via MLX) with a Pydantic schema describing the expected output: `tipo_fondale`, `granulometria`, `presenza_rocce`, `presenza_ciottoli`, `copertura_algale`, `taxa_algali`, `elementi_antropici`, `fauna`. The schema is in Italian to match the GT vocabulary, but the prompt instructs the model to *reason* in English before emitting Italian enum tokens (Gemma 4's English reasoning is more reliable than its Italian).

**Stage 3 — Deduplication and rendering.**
Adjacent frames with the same categorical signature get merged. We use DINOv3 ViT-S/16 embeddings (via PyTorch + MPS) as a secondary signal: two frames merge only when their VLM-assigned categories agree *AND* their cosine similarity exceeds a tuned threshold. The merged segments are rendered into a Markdown report using fixed Italian templates (anti-hallucination — the natural language is templated, only the structured fields come from the model).

Supporting modules: `sync.py` (Deep Trekker log + GPX → per-frame telemetry), `enhance.py` (gray-world + CLAHE + unsharp mask preprocessing before both the VLM and the embedder), `schema.py` (Pydantic enums for the controlled vocabulary), `eval/` (ground-truth annotation harness against 93 expert captions, with F1 / Jaccard comparison across pipeline configurations).

---

## 3. Key decisions and why

Each subsection below is structured as **Problem → What we tried → What worked → Why it mattered**.

### 3.1 Local-only constraint shapes the entire stack

**Problem.** No paid APIs allowed. Most "easy" VLM solutions (GPT-4V, Claude Sonnet, Gemini) were ruled out.

**What we tried.** First passed: Qwen 2.5-VL-7B (mlx-vlm port). Then Gemma 4 E4B, also 4-bit quantized via MLX. Quick qualitative comparison on a handful of frames showed Gemma 4 producing cleaner schema-compliant output and getting fewer obvious categorical errors.

**What worked.** Gemma 4 E4B as the default VLM. Locked in as production, Qwen kept as an option behind a `--model` flag.

**Why it mattered.** This decision shaped everything downstream — output token length budgets, the choice to use MLX over PyTorch for the VLM, the structured-output strategy. It also produced the writeup angle: *"we did a real comparison of open-weight VLMs on a domain-specific task rather than defaulting to a closed API."*

**Honest caveat.** This comparison was qualitative, not quantitative. A proper Qwen-vs-Gemma ablation against the GT would have been better. We didn't do it because (a) it would have doubled eval compute time, (b) by mid-project Qwen3-VL had been released and the meaningful comparison would be Gemma 4 vs Qwen3-VL anyway — a thesis-extension question, not a course-project one.

### 3.2 The output schema is anti-hallucination, not pro-richness

**Problem.** VLMs love to hallucinate. Asking Gemma 4 to "write a description of this frame in Italian" produced fluent text that often described things not in the image — phantom organisms, invented coral species, plausible-but-fictitious archaeological objects.

**What we tried.** First version: prompt with a Pydantic schema, parse JSON output, render the JSON into a fixed Italian template. The VLM only fills categorical fields; the natural-language report is generated from templates.

**What worked.** Hallucinations dropped sharply. The captions in the final report are *less rich* than what the VLM would freely generate — they don't speculate about coral species or write evocative prose — but every claim is grounded in a schema field that came from looking at the image.

**Why it mattered.** Forcing structure trades off fluency for trust. For an *inspection report* that goes to a port authority, trust is the right tradeoff. For a creative captioning system it would be the wrong one.

### 3.3 Candidate scoring needs fine-grained temporal buckets, not per-minute top-K

**Problem.** Early pipeline used `top_per_minute=6` for candidate selection. On Video 2, a dense rocky breakwater zone at t=43–50s wasn't appearing in the report. We initially blamed: the VLM ("it's mislabeling rocks as sand"), then DINOv2 embeddings ("they're merging the rocky zone into adjacent gravel"), then the depth filter ("shallow content is being dropped").

**What we tried.** Added camera-pitch-aware depth filtering. Switched from DINOv2 to DINOv3. Tightened the merge rule. Lowered the embedding threshold. None of these fixed it.

**What worked.** A diagnostic script printed raw sharpness and speed scores around t=50. The boulder frames *were sampled* — they just lost the top-6 cut to slightly sharper neighbors at t=27-36s. The bucket window was the issue: with `top_per_minute=6`, six photographically clean still-water frames at the start of a minute crowded out everything informative in the rest of the same minute. Switching to `bucket_seconds=20, top_per_bucket=3` enforced uniform temporal coverage and immediately surfaced the rocky zone.

**Why it mattered.** This is a real CV finding worth documenting: photographic quality scoring rewards stillness and sharpness, which are correlated with "boring" content. The fix is one line of code (`int(c.t_sec // 60)` → `int(c.t_sec // 20)`) but it's principled — sub-minute bucketing trades a small increase in candidate count for genuine coverage uniformity.

**Also a meta-lesson.** We spent ~90 minutes proposing speculative fixes before running the diagnostic that took 30 seconds and identified the actual bug. Instrument first, hypothesize second.

### 3.4 Underwater color cast hurts both the VLM and the embeddings

**Problem.** Adjacent frames at constant depth flipping between `sabbioso` and `ghiaioso` labels. DINOv2 cosine similarities for visually-identical underwater frames coming in at ~0.59 instead of the >0.95 we'd expect for terrestrial scenes. Both symptoms of the same underlying issue: blue-green color cast compresses texture and color information into a narrow band of the model's feature space.

**What we tried.** A small `enhance.py` module: per-channel gray-world white balance, CLAHE on the L channel of LAB color space, light unsharp mask. Applied to each candidate frame *before* both the VLM call and the DINOv3 embedding.

**What worked.** Quantitatively measured on the 20-figure GT subset: aggregate F1 improved from 0.39 to 0.44. The largest single-metric gains were on rock/cobble presence (+18 and +13 F1 points absolute). On Video 3 specifically, the categorical flip-flops between sabbioso/ghiaioso largely disappeared.

**Why it mattered.** This is one of two interventions in the project with a measurable, reproducible effect. The enhancement runs in ~50ms per frame (negligible) and doesn't require any model retraining. The technique itself is classical (predates deep learning by decades), but the *finding* — that classical preprocessing can move foundation-model output by 10+ F1 points on out-of-distribution data — is the kind of result worth reporting.

**Honest caveat.** It's not strictly an improvement everywhere. Two of the 20 GT frames regressed (the VLM became *over-confident* in seeing gravel on actually-sandy frames). The eval surfaces this as a directional bias flip rather than a strict win.

### 3.5 Embedding-based dedup needs AND, not OR

**Problem.** Adjacent frames with the same categorical signature getting un-merged because the VLM gave them slightly different `tipo_fondale` values; conversely, visually-distinct findings (e.g., a Corallinales-encrusted rock zone next to a plain gravel zone) getting merged because DINOv3 said they looked similar.

**What we tried.** First rule: merge if (categorical match) OR (embedding similarity > threshold). This over-merged visually-similar but categorically-distinct content. Switched to AND: merge only if both signals agree. With AND, the threshold became a *secondary* check rather than a primary driver — set fairly low (0.50 with DINOv3 on underwater data, accounting for the compression mentioned in §3.4).

**What worked.** AND-rule + threshold 0.50 preserved the Corallinales/rocky finding in Video 2 while still collapsing the redundant gravel sequence.

**Why it mattered.** This is a clean design decision worth flagging: the OR rule was *more aggressive merging*, AND is *more conservative*. When both signals (categorical agreement + visual similarity) must align, false merges go down at the cost of slight over-segmentation. For an inspection report where missing a finding is worse than reporting two figures of the same thing, AND is the right default.

### 3.6 DINOv2 vs DINOv3 on out-of-distribution data

**Problem.** DINOv2 ViT-S/16 produced compressed similarity ranges on underwater frames. Adjacent visually-identical gravel frames returned cosine similarity ~0.59. Adjacent gravel vs. distinctly different rocky-with-Corallinales zone returned 0.71. The model considered the rocky zone *more similar* to gravel than two consecutive gravel frames were to each other — making any threshold useless.

**What we tried.** Upgraded to DINOv3 ViT-S/16 (Meta, August 2025) when access became available. The pip install path was painful (mlx-image required an old scipy with no Python 3.13 wheel and Fortran compilation), so we switched to the PyTorch + transformers + MPS path.

**What worked.** Adjacent gravel similarity rose from 0.59 (DINOv2) to 0.67 (DINOv3). Cross-scene discrimination didn't improve as much as hoped (gravel vs. rocky-Corallinales still ~0.71 with DINOv3). The improvement is real but smaller than the marketing materials suggested.

**Why it mattered.** Two things, both worth saying in the writeup:
1. Foundation models pretrained on terrestrial imagery don't fully transfer to underwater scenes. The similarity ranges compress because color cast + low contrast + texture homogeneity reduce the effective dimensionality of useful features.
2. DINOv3 is a real improvement but not the silver bullet. The remaining discrimination gap is what motivated us to enforce categorical agreement (§3.5) rather than rely on embeddings alone.

### 3.7 Evaluation: GT as floor, not ceiling

**Problem.** Halfway through the project, we needed to measure progress with numbers, not anecdotes. The natural baseline is the 93-figure expert document — but that document is a *rapid expert pass*, not an exhaustive labeling. The human picked ~10 representative frames per video; our pipeline produces ~15-30 per video. Computing recall against the GT would penalize us for finding *more* than the human did.

**What we tried.** Reframed the eval philosophy: GT is the floor (does the pipeline match expert-flagged findings?), not the ceiling (the pipeline can and should find more). Stratified-sampled 20 GT figures across all videos. Encoded each one into structured Pydantic fields by hand. Built `eval/run_eval.py` (runs VLM on each GT image with current pipeline config) and `eval/compare.py` (loads multiple config results, prints per-metric F1/Jaccard table).

**What worked.** First baseline run: aggregate F1 ≈ 0.40 across 8 metrics. After color correction (§3.4): 0.50. Measurable, reproducible, ablatable.

**Why it mattered.** Most importantly: it converted every subsequent change from "this looks better" to "this moved metric X by Y points." Two specific metric-design choices we made and should explain in the writeup:
- `taxa_algali` Jaccard treats `[]` and `['non_identificata']` as equivalent (the model emits `non_identificata` when it sees algae but can't ID the taxon; the human encodes the same observation as `[]`). Without this normalization, the metric was artificially zero.
- `granulometria` macro-F1 is computed only over entries where the GT specified a grain-size category. The human leaves it null for most sand-dominated frames; the VLM commits to a value for every frame. Computing F1 over all entries gave 0.000 (~75% of entries are null in GT, where every prediction is "wrong"). Restricting to specified-only entries surfaces the real signal.

After this draft, the eval was extended to all 93 GT figures. Results to be added when complete.

### 3.8 Things that didn't work (or worked less than expected)

A real project log includes the things that didn't pay off:

- **Italian-only prompting.** Early prompts asked the VLM to reason in Italian. Gemma 4 defaults to safer / more generic categories when reasoning in Italian (sand-by-default). Switching to "reason in English, emit Italian tokens" recovered specificity but introduced over-specification in the other direction. Net positive but not a clean win.
- **DINOv2 → DINOv3 transition.** Real, but smaller than expected (§3.6).
- **Per-frame depth filtering only.** Adding camera-pitch refinement (`if depth < 0.8m AND camera_pitch > -10°: skip`) turned out to be unnecessary in our dataset — the depth filter alone was correctly excluding water-surface artifacts, and shallow inspection content always had cam_pitch ≲ -30°. We left the camera-pitch check in for robustness but it didn't change behavior on Messina footage.
- **Bigger embedding models.** DINOv3 ViT-S/16 (~22M params) was sufficient. We considered ViT-L (~300M) and didn't run it — the marginal benefit didn't justify 14× the latency for what is, in this pipeline, a secondary signal.

---

## 4. Known limitations

Stated honestly, in rough order of how much they hurt the system.

**4.1 Multi-class substrate F1 is mediocre.** `tipo_fondale` macro-F1 sits around 0.30 even after enhancement. The model resolves sand-vs-gravel correctly on clear-cut frames but flips on borderline cases, especially the `sabbioso_misto_a_ghiaia` middle category. A bigger VLM (e.g., Qwen3-VL) would probably help here.

**4.2 Fine granulometric discrimination is essentially zero.** The five GT entries that specified grain size (`ciottoli_piccoli` / `ciottoli_medi` / `grossolana`) — the model got *none* right. This is plausibly a capability ceiling for zero-shot VLMs on underwater texture, not a fixable bug.

**4.3 The pipeline ignores absolute camera distance to substrate.** "Ciottoli" (cobbles) vs "ghiaia" (gravel) is partly a function of *real-world particle size*, which requires scale calibration. Our ROV has a 30cm laser baseline that could provide ground-truth scale per frame; we don't currently use it.

**4.4 No temporal smoothing of categorical predictions.** Each frame is analyzed independently. A categorical flip between adjacent frames at the same substrate is a model error, not a real scene change. A simple temporal majority filter (or V-JEPA, see §5) would smooth these.

**4.5 Eval sample size.** 93 entries is reasonable but not large. Single-frame errors can move per-class F1 by 5+ points. Coverage of rare categories (`Laminariales`, `mollusco_bivalve`, fauna in general) is too sparse for confident per-class reporting.

**4.6 Dependency fragility.** Python 3.13 + Apple Silicon + multiple ML frameworks (mlx-vlm, transformers, torch, mlx) means installs are not reproducible across machines without effort. The journey from "let's add DINOv3" to a working `embed.py` took roughly an hour of dependency-resolution debugging.

---

## 5. Next steps & open directions

Three directions, ordered by impact:

### 5.1 Concept-prompted substrate segmentation with SAM 3 (most novel)

The current pipeline produces *frame-level* categorical labels: "this frame shows fondale ghiaioso." A natural extension is *pixel-level* substrate maps: "this region of the frame is gravel, this region is rock with algal cover, this region is sand." SAM 3 (Meta, November 2025) is built for exactly this — it accepts text prompts ("rocks covered in algae") and returns segmentation masks for every matching region.

The Mediterranean marine project FathomNet is already using SAM 3 for underwater object segmentation, so there's domain precedent.

**Concrete starting points for a teammate:**
- Repo: `facebookresearch/sam3` (open weights, ~840M params, ~3.4 GB on disk, requires GPU inference; PyTorch + MPS on M4 should work but be slow)
- No MLX port yet — PyTorch only as of the latest check
- Suggested first experiment: run SAM 3 with prompt `"rocks covered in algae"` on the 25 frames in our GT where `presenza_rocce: true` AND `copertura_algale: diffusa`. Compare the resulting masks against visual inspection. Does the model find them all? Are the masks tight?
- Suggested integration: a hybrid mode where the existing VLM pipeline runs as today, and SAM 3 runs *only* when the VLM flags unusual content (anthropogenic items, named taxa, fauna). This keeps the latency budget reasonable while adding pixel-level evidence for the interesting findings.
- Output potential: per-frame substrate-coverage percentages (e.g., "63% gravel, 28% algae-covered rock, 9% sand") which are closer to what marine engineers actually want from inspection reports than categorical labels.

This is genuinely novel work and could be the core of a thesis chapter.

### 5.2 Temporal reasoning with V-JEPA (most principled)

V-JEPA (Meta) reasons over windows of video frames rather than single frames. Two things this would help with:

- **Categorical smoothing.** A single-frame model flip-flops on borderline substrate (see §4.4); a video model can use the surrounding ~5 seconds to commit to one category.
- **Change-point detection for representative frame selection.** Currently our frame selection is per-frame quality + temporal bucketing. A video model could detect *scene boundaries* — where the substrate or environment genuinely changes — and place representative frames at those boundaries instead of on a fixed temporal grid.

**Concrete starting points:**
- Repo: `facebookresearch/jepa`
- Smaller computational footprint than SAM 3 but trickier to integrate (works on video clips, not single frames)
- Suggested first experiment: run V-JEPA on each video's full timeline, extract change-points, and check whether they correlate with the timestamps where the human expert picked GT frames. If they do, V-JEPA-driven frame selection is a credible upgrade to our classical-CV scorer.

### 5.3 Quantitative comparison vs. Qwen3-VL (most measurable)

We chose Gemma 4 qualitatively early in the project. By submission time, Qwen3-VL had been released with a 4B variant (`mlx-community/Qwen3-VL-4B-Instruct-4bit`) and a much larger MoE variant. Running the full 93-figure eval on Qwen3-VL would produce a head-to-head comparison number.

This is the lowest-novelty extension but the highest-clarity result. Two `run_eval.py` commands, one new `compare.py` invocation, and you have a defensible "which model is better for this domain" finding.



---

## 6. Repo navigation cheat-sheet for a new collaborator

```
rov-inspect/
├── src/rov_inspect/        # the pipeline itself
│   ├── schema.py           # Pydantic enums (the controlled vocabulary)
│   ├── sync.py             # telemetry + GPX synchronization
│   ├── candidates.py       # frame scoring + bucketing (see §3.3)
│   ├── enhance.py          # underwater color correction (see §3.4)
│   ├── vlm_local.py        # MLX-hosted Gemma 4 caller
│   ├── vlm_common.py       # prompt text + schema docs
│   ├── embed.py            # DINOv3 via PyTorch MPS (see §3.6)
│   ├── segment.py          # AND-rule dedup (see §3.5)
│   ├── render.py           # Markdown templates (anti-hallucination)
│   └── pipeline.py         # the orchestrator
├── scripts/run_pipeline.py # CLI entry point
├── eval/                   # the eval harness (see §3.7)
│   ├── extract_gt.py
│   ├── gt_annotations_subset.json  # 93 hand-coded GT entries
│   ├── run_eval.py
│   └── compare.py
├── CLAUDE.md               # working notes from build sessions
├── NOTES.md                # design philosophy
└── README.md               # user-facing docs
```

For a teammate stepping in: start by reading this log, then `CLAUDE.md`, then the `eval/` directory. The pipeline orchestrator (`src/rov_inspect/pipeline.py`) is the right place to understand how the pieces fit together; the schema (`src/rov_inspect/schema.py`) is the right place to understand the output contract.

---

*Last updated: {13/05/2026}. Project state at this log: all of §1–§4 implemented and committed, eval results on 20-figure subset reproduced, 93-figure eval running at time of writing. SAM 3 / V-JEPA extensions are not implemented.*
