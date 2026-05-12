# Evaluation

Ground-truth and harness for measuring pipeline quality against the
human-authored figures in `Rov_Immagine.docx`.

## Workflow

1. **Extract** GT images + caption template from the docx
   (one-time; reproducible):

   ```
   python eval/extract_gt.py --docx ../Rov_Immagine.docx
   ```

   Produces:
   - `eval/gt_images/figure_NN.jpg` — 93 extracted images (gitignored)
   - `eval/gt_annotations_template.json` — all 93 entries, `expected`
     fields empty
   - `eval/gt_annotations_subset.json` — a 20-figure stratified sample
     spanning every video and the major content types

2. **Annotate the subset.** Open
   `eval/gt_annotations_subset.json` and fill in the `expected` block
   for each entry based on the `original_caption` (and the image if
   the caption is ambiguous). Use schema enum values verbatim
   (`sabbioso`, `rocce_antropiche`, …). Leave defaults (`null`, `[]`,
   `false`) where the caption does not mention a field — under-specified
   is better than over-specified for evaluation fairness.

   Only the subset needs to be filled in by hand; the full template
   is kept so a larger eval set can be added later without re-extracting.

3. **Run eval** on each pipeline configuration (script lands in
   step 3 of this work):

   ```
   python eval/run_eval.py --config baseline --subset
   ```

   Writes per-config outputs under `eval/results/<config>/` (gitignored).

4. **Compare** configurations (script lands in step 3):

   ```
   python eval/compare.py baseline enhanced ...
   ```

## File layout

```
eval/
├── extract_gt.py                  # one-time GT extractor (tracked)
├── gt_annotations_template.json   # all 93 entries, defaults (tracked)
├── gt_annotations_subset.json     # 20-figure annotation target (tracked)
├── gt_images/                     # extracted JPEGs (gitignored)
├── results/                       # per-config eval outputs (gitignored)
└── README.md
```

`gt_images/` is gitignored because the images derive from
`Rov_Immagine.docx`, which is private survey data and lives outside
the repo. The two JSON files are tracked — they are reproducible from
the docx and form the evaluation contract.
