"""
Merge adjacent candidates that look at "the same thing".

v0.2: simple categorical-equality rule. Adjacent candidates with
the same coarse signature are merged into one segment; keep the
highest-quality frame as representative.

Future: plug in DINOv2 / CLIP embedding similarity for finer
content-based dedup (e.g. same anthropic items but different zones
of breakwater).
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from .candidates import FrameCandidate
from .embed import cosine_similarity
from .schema import FrameAnalysis


@dataclass
class Segment:
    candidates: list[FrameCandidate]
    analyses: list[FrameAnalysis]

    @property
    def representative(self) -> tuple[FrameCandidate, FrameAnalysis]:
        best_i = max(
            range(len(self.candidates)),
            key=lambda i: self.candidates[i].score * self.analyses[i].qualita_visuale,
        )
        return self.candidates[best_i], self.analyses[best_i]

    @property
    def t_start(self) -> float:
        return self.candidates[0].t_sec

    @property
    def t_end(self) -> float:
        return self.candidates[-1].t_sec


def _signature(a: FrameAnalysis) -> tuple:
    return (
        a.tipo_fondale.value,
        a.copertura_algale.value,
        frozenset(e.value for e in a.elementi_antropici),
        frozenset(f.tipo.value for f in a.fauna),
    )


def merge_segments(
    candidates: list[FrameCandidate],
    analyses: list[FrameAnalysis],
    max_gap_sec: float = 30.0,
    embeddings: list[np.ndarray] | None = None,
    embedding_threshold: float = 0.92,
) -> list[Segment]:
    """Merge temporally-adjacent candidates that depict the same content.

    Merge rule: adjacent frames are joined iff the time gap is within
    ``max_gap_sec`` AND their categorical signatures match. When
    ``embeddings`` is provided, DINOv3 cosine similarity acts as a
    secondary verification signal — both the categorical signature AND
    the embedding similarity must agree (cosine > ``embedding_threshold``)
    before merging. Embeddings never override a categorical disagreement;
    they only guard against the VLM emitting identical categorical labels
    for visually different scenes. When ``embeddings`` is ``None``, the
    function falls back to categorical-only merging.
    """
    if not candidates:
        return []
    segments: list[Segment] = []
    cur_c = [candidates[0]]
    cur_a = [analyses[0]]
    cur_sig = _signature(analyses[0])
    cur_emb = embeddings[0] if embeddings is not None else None

    rest = zip(candidates[1:], analyses[1:])
    for idx, (c, a) in enumerate(rest, start=1):
        sig = _signature(a)
        gap = c.t_sec - cur_c[-1].t_sec
        emb = embeddings[idx] if embeddings is not None else None

        sig_match = sig == cur_sig
        if embeddings is None:
            content_match = sig_match
        else:
            emb_match = (
                emb is not None
                and cur_emb is not None
                and cosine_similarity(cur_emb, emb) > embedding_threshold
            )
            content_match = sig_match and emb_match
        if content_match and gap <= max_gap_sec:
            cur_c.append(c)
            cur_a.append(a)
        else:
            segments.append(Segment(cur_c, cur_a))
            cur_c, cur_a, cur_sig, cur_emb = [c], [a], sig, emb
    segments.append(Segment(cur_c, cur_a))
    return segments
