# Project log

Decisions, framing, and open questions. This file replaces the
top-level `NOTES.md` from earlier iterations of the repo and is the
place to record context that doesn't belong in the code or CLAUDE.md.

## Project framing

## On the ground truth

The Rov_Immagine.docx annotations represent a single expert's rapid review
of the footage. The expert selected representative frames and wrote brief
descriptions, prioritizing speed over exhaustive coverage.

This system is positioned as an *AI-assisted augmentation*, not an imitator:

- Provides consistent coverage across the full video, not just selected moments
- Adds geo-anchored metadata (lat/lon, depth, timestamp) the human did not have
- Surfaces findings the expert may have skipped under time pressure
- Maintains audit-traceable structured output (Pydantic-validated)

For evaluation:
- Treat GT agreement as a *recall lower bound* (does the AI find what the human found?)
- Report *coverage density* (findings per minute) as a separate axis
- Do NOT compute precision against GT — the AI is expected to find more, not less

## SAM 3 spatial verification

SAM 3 is wired in as an optional post-processing layer on top of a
completed pipeline run, not as part of the main pipeline. The VLM
remains responsible for *what* the frame contains (categorical
classification); SAM 3 adds *where* — pixel-level outlines for each
finding the VLM named.

### Design decisions

- **Report-driven prompt selection.** Rather than running every prompt
  on every frame, the script parses the rendered `Descrizione` line of
  each figure, matches Italian keywords against
  `DESCRIZIONE_KEYWORDS_TO_PROMPT` in
  `src/rov_inspect/segment_sam3.py`, and runs only the SAM 3 prompts
  the VLM hinted at. This cuts per-figure inference 3–5× and avoids
  prompts that have no chance of grounding (e.g. running "fishing net"
  on a pure-sand frame).
- **Outline-only rendering.** The earliest iteration used alpha-blended
  colour fills. They obscured the underlying frame and made
  over-grounding hard to assess — a 90%-area mask of "rocks" looked
  identical to a tight outline around a real boulder. Switching to
  contour-only rendering (`cv2.drawContours` at thickness 2–3) keeps
  the original image fully visible inside each region; legitimate
  detections show up as tight contours and over-grounding shows up as
  obvious frame-spanning blobs.
- **Hardcoded to Video 3 for the current demo.** Generalisation to
  arbitrary run directories is straightforward (the `--run-dir`
  argument is already parsed) but currently ignored to keep the demo
  artifact deterministic.

### Prompt set evolution

The prompt set changed several times based on empirical findings on
Messina footage:

1. **Substrate prompts (`sand`, `rocks`, `boulders`).** Smoke-tested
   first via `scripts/sam3_smoke.py` on a single frame
   (`out/video3_enhanced/sam3_smoke/`). Each prompt grounded 25–35% of
   the frame with heavy mutual overlap; under colour fills this looked
   like noise. Dropped.
2. **Discrete-finding only (`fishing net`, `pipe`, `tire`, `debris`,
   `sea urchin or starfish`).** SAM 3 reliably grounds discrete
   objects, and the VLM had a clear vocabulary for them. Adopted as
   the second iteration.
3. **Fauna removed.** The Messina data has very few real fauna
   occurrences. On the three Video 3 figures where the VLM said
   `riccio di mare`, SAM 3 returned 0% coverage on `fish` / `sea
   urchin` prompts — i.e. the urchins almost certainly weren't there
   (VLM hallucination), and SAM 3 was correctly silent. Keeping the
   prompt added cost without information.
4. **Substrate re-enabled under outline-only rendering.** Once fills
   were replaced with contours, "rocks" and "vegetation" became
   useful again: the user can visually judge whether a big contour is
   plausible. `rocce` and `ciottoli` were split into two classes so
   the VLM's distinct vocabulary mapped cleanly to two prompts.

### Empirical hit rates on Video 3 (39 figures processed)

Using the current prompt set:

| prompt       | figures fired (>0%) | typical coverage when fired |
|--------------|---------------------|------------------------------|
| `rocks`      | 10 / 39             | 5–93%, with a long tail      |
| `vegetation` | 4 / 39              | 0.2–3%                       |
| `cobbles`    | 0 / 39              | —                            |

`cobbles` never grounded anything on this dataset; the Italian keyword
`ciottoli` may be better routed to the `rocks` prompt in a future
iteration. The high-end "rocks" coverages (80–93% on figs 9/10/14/15)
look plausible on inspection — those frames really are mostly rocky
substrate.

### Files

- `src/rov_inspect/segment_sam3.py` — model loader, `segment_frame()`,
  `FINDING_PROMPTS`, `DESCRIZIONE_KEYWORDS_TO_PROMPT`.
- `scripts/run_sam3_postprocess.py` — end-to-end post-processor,
  priority-combined label maps, outline rendering, consolidated
  Markdown writer.
- `scripts/sam3_smoke.py` — single-image multi-prompt smoke test, used
  to triage prompt phrasings before committing them.
- `demo/video3_enhanced/sam3_verification/` — one committed example
  output (figure 18 of Video 3).