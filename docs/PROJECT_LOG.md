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