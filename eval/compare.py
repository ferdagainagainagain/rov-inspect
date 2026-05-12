#!/usr/bin/env python3
"""Compare multiple eval result files by per-field F1 / Jaccard.

Usage:
    python eval/compare.py eval/results/baseline.json eval/results/enhanced.json ...

Metric definitions (see CLAUDE.md / eval/README.md):
  - tipo_fondale, copertura_algale       -> macro-F1 over enum classes
  - granulometria                        -> macro-F1 over non-null classes
                                            (null vs null = agreement)
  - presenza_rocce, presenza_ciottoli    -> binary F1
  - taxa_algali, elementi_antropici      -> mean Jaccard over label sets
  - fauna                                -> mean Jaccard over tipo strings
                                            (specie ignored)
  - Weighted aggregate                   -> equal-weight mean of the 8.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from statistics import mean


# ── metric helpers ────────────────────────────────────────────────────

def _binary_f1(tp: int, fp: int, fn: int) -> float:
    if tp == 0 and (fp > 0 or fn > 0):
        return 0.0
    if tp == 0 and fp == 0 and fn == 0:
        return 1.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def macro_f1(pairs: list[tuple]) -> float:
    """Macro-F1 over (expected, predicted) pairs. Skips pairs where both
    sides are None (counted as separate "agreement" handling by callers).
    """
    classes = {e for e, _ in pairs if e is not None} | {p for _, p in pairs if p is not None}
    if not classes:
        return 1.0
    f1s = []
    for cls in classes:
        tp = sum(1 for e, p in pairs if e == cls and p == cls)
        fp = sum(1 for e, p in pairs if e != cls and p == cls)
        fn = sum(1 for e, p in pairs if e == cls and p != cls)
        f1s.append(_binary_f1(tp, fp, fn))
    return mean(f1s) if f1s else 1.0


def binary_field_f1(pairs: list[tuple[bool | None, bool | None]]) -> float:
    """Binary F1, treating None as False (under-specified GT = absent)."""
    tp = sum(1 for e, p in pairs if bool(e) and bool(p))
    fp = sum(1 for e, p in pairs if not bool(e) and bool(p))
    fn = sum(1 for e, p in pairs if bool(e) and not bool(p))
    return _binary_f1(tp, fp, fn)


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def mean_jaccard(pairs: list[tuple[set, set]]) -> float:
    if not pairs:
        return 1.0
    return mean(jaccard(a, b) for a, b in pairs)


GRANULOMETRIA_MIN_SAMPLES = 3


def granulometria_f1(pairs: list[tuple]) -> float | None:
    """Macro-F1 restricted to entries where GT is non-null.

    Granulometria is null in the GT when the human caption didn't
    specify grain size — those rows are 'metric does not apply', not
    failures, so they're dropped before computing F1. If fewer than
    GRANULOMETRIA_MIN_SAMPLES non-null GT rows remain, the metric is
    reported as N/A (returns None).
    """
    filtered = [(e, p) for e, p in pairs if e is not None]
    if len(filtered) < GRANULOMETRIA_MIN_SAMPLES:
        return None
    return macro_f1(filtered)


def granulometria_n(pairs: list[tuple]) -> int:
    return sum(1 for e, _ in pairs if e is not None)


# ── data extraction ───────────────────────────────────────────────────

def _fauna_tipi(items: list | None) -> set[str]:
    if not items:
        return set()
    return {it.get("tipo") for it in items if isinstance(it, dict) and it.get("tipo")}


def _label_set(items: list | None) -> set[str]:
    if not items:
        return set()
    return set(items)


def _taxa_set(items: list | None) -> set[str]:
    """Label set for taxa_algali, treating ['non_identificata'] as []."""
    s = _label_set(items)
    if s == {"non_identificata"}:
        return set()
    return s


def extract_pairs(entries: list[dict]) -> dict:
    """Pull (expected, predicted) tuples per field, skipping entries
    with no prediction."""
    ok = [e for e in entries if e.get("predicted") is not None]
    return {
        "tipo_fondale": [(e["expected"].get("tipo_fondale"),
                          e["predicted"].get("tipo_fondale")) for e in ok],
        "granulometria": [(e["expected"].get("granulometria"),
                           e["predicted"].get("granulometria")) for e in ok],
        "presenza_rocce": [(e["expected"].get("presenza_rocce"),
                            e["predicted"].get("presenza_rocce")) for e in ok],
        "presenza_ciottoli": [(e["expected"].get("presenza_ciottoli"),
                               e["predicted"].get("presenza_ciottoli")) for e in ok],
        "copertura_algale": [(e["expected"].get("copertura_algale"),
                              e["predicted"].get("copertura_algale")) for e in ok],
        "taxa_algali": [(_taxa_set(e["expected"].get("taxa_algali")),
                         _taxa_set(e["predicted"].get("taxa_algali"))) for e in ok],
        "elementi_antropici": [(_label_set(e["expected"].get("elementi_antropici")),
                                _label_set(e["predicted"].get("elementi_antropici"))) for e in ok],
        "fauna": [(_fauna_tipi(e["expected"].get("fauna")),
                   _fauna_tipi(e["predicted"].get("fauna"))) for e in ok],
        "_n_ok": len(ok),
        "_n_total": len(entries),
    }


METRICS = [
    ("tipo_fondale (macro-F1)",       "tipo_fondale",       macro_f1),
    ("granulometria (macro-F1)",      "granulometria",      granulometria_f1),
    ("presenza_rocce (F1)",           "presenza_rocce",     binary_field_f1),
    ("presenza_ciottoli (F1)",        "presenza_ciottoli",  binary_field_f1),
    ("copertura_algale (macro-F1)",   "copertura_algale",   macro_f1),
    ("taxa_algali (Jaccard)",         "taxa_algali",        mean_jaccard),
    ("elementi_antropici (Jaccard)",  "elementi_antropici", mean_jaccard),
    ("fauna (Jaccard)",               "fauna",              mean_jaccard),
]


def build_labels(pairs_list: list[dict]) -> list[str]:
    """Return display labels in METRICS order. Granulometria's label
    embeds the sample count so the smaller-n is visible. We use the
    max non-null count across configs so the label is stable when
    predictions differ slightly between configs."""
    gran_n = max(granulometria_n(pp["granulometria"]) for pp in pairs_list) if pairs_list else 0
    labels: list[str] = []
    for label, key, _ in METRICS:
        if key == "granulometria":
            labels.append(f"granulometria (macro-F1, n={gran_n})")
        else:
            labels.append(label)
    return labels


def compute_scores(pairs: dict, labels: list[str]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for (_, key, fn), label in zip(METRICS, labels):
        out[label] = fn(pairs[key])
    return out


# ── disagreement breakdown ────────────────────────────────────────────

def collect_disagreements(entries: list[dict]) -> list[tuple[int, str, object, object]]:
    rows: list[tuple[int, str, object, object]] = []
    for e in entries:
        pred = e.get("predicted")
        if pred is None:
            continue
        exp = e["expected"]
        fig = e["figure_id"]
        for f in ("tipo_fondale", "granulometria", "copertura_algale"):
            if exp.get(f) != pred.get(f):
                rows.append((fig, f, exp.get(f), pred.get(f)))
        for f in ("presenza_rocce", "presenza_ciottoli"):
            if bool(exp.get(f)) != bool(pred.get(f)):
                rows.append((fig, f, exp.get(f), pred.get(f)))
        for f in ("taxa_algali", "elementi_antropici"):
            if _label_set(exp.get(f)) != _label_set(pred.get(f)):
                rows.append((fig, f, exp.get(f), pred.get(f)))
        if _fauna_tipi(exp.get("fauna")) != _fauna_tipi(pred.get("fauna")):
            rows.append((fig, "fauna", exp.get("fauna"), pred.get("fauna")))
    return rows


# ── rendering ─────────────────────────────────────────────────────────

def render_table(
    configs: list[str],
    scores: list[dict[str, float | None]],
    labels: list[str],
) -> str:
    label_w = max(len(label) for label in labels) + 2
    col_w = max(max(len(c) for c in configs) + 2, 12)

    header = f"{'Metric':<{label_w}}" + "".join(f"{c:<{col_w}}" for c in configs)
    sep = "─" * (label_w + col_w * len(configs))
    lines = [header, sep]

    for label in labels:
        row_scores = [s[label] for s in scores]
        numeric = [v for v in row_scores if v is not None]
        best = max(numeric) if numeric else None
        cells = []
        for v in row_scores:
            if v is None:
                cells.append("N/A (insufficient samples)".ljust(col_w))
            else:
                star = "*" if best is not None and abs(v - best) < 1e-9 else " "
                cells.append(f"{v:.3f}{star}".ljust(col_w))
        lines.append(f"{label:<{label_w}}" + "".join(cells))

    aggregates = [
        mean([v for v in s.values() if v is not None]) if any(v is not None for v in s.values()) else 0.0
        for s in scores
    ]
    best_agg = max(aggregates)
    agg_cells = []
    for v in aggregates:
        star = "*" if abs(v - best_agg) < 1e-9 else " "
        agg_cells.append(f"{v:.3f}{star}".ljust(col_w))
    lines.append(sep)
    lines.append(f"{'Weighted aggregate':<{label_w}}" + "".join(agg_cells))
    return "\n".join(lines)


def render_disagreements(name: str, rows: list[tuple], cap: int = 20) -> str:
    if not rows:
        return f"[{name}] no disagreements"
    head = f"[{name}] disagreements ({len(rows)} total, showing up to {cap}):"
    lines = [head]
    for fig, field, exp, pred in rows[:cap]:
        lines.append(f"  fig {fig:>3}  {field:<22}  expected={exp!r:<30}  predicted={pred!r}")
    if len(rows) > cap:
        lines.append(f"  … {len(rows) - cap} more")
    return "\n".join(lines)


# ── main ──────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("results", type=Path, nargs="+", help="One or more eval result JSONs.")
    p.add_argument("--disagreement-cap", type=int, default=20)
    args = p.parse_args()

    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in args.results]
    configs = [pl["config"]["name"] for pl in payloads]
    entries_by_cfg = [pl["entries"] for pl in payloads]

    pairs = [extract_pairs(es) for es in entries_by_cfg]
    labels = build_labels(pairs)
    scores = [compute_scores(pp, labels) for pp in pairs]

    print()
    for cfg, pp in zip(configs, pairs):
        print(
            f"[{cfg}] {pp['_n_ok']}/{pp['_n_total']} predictions parsed"
        )
    print()
    print(render_table(configs, scores, labels))
    print()
    for cfg, es in zip(configs, entries_by_cfg):
        rows = collect_disagreements(es)
        print(render_disagreements(cfg, rows, cap=args.disagreement_cap))
        print()


if __name__ == "__main__":
    main()
