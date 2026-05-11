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

from .candidates import FrameCandidate
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
) -> list[Segment]:
    if not candidates:
        return []
    segments: list[Segment] = []
    cur_c = [candidates[0]]
    cur_a = [analyses[0]]
    cur_sig = _signature(analyses[0])

    for c, a in zip(candidates[1:], analyses[1:]):
        sig = _signature(a)
        gap = c.t_sec - cur_c[-1].t_sec
        if sig == cur_sig and gap <= max_gap_sec:
            cur_c.append(c)
            cur_a.append(a)
        else:
            segments.append(Segment(cur_c, cur_a))
            cur_c, cur_a, cur_sig = [c], [a], sig
    segments.append(Segment(cur_c, cur_a))
    return segments
