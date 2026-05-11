"""
Render structured analyses into Italian inspection captions and reports.

Templated for fidelity to the GT style: the VLM returns categorical
fields, this module renders them into prose. Prevents the VLM from
inventing taxa or items not in the schema and keeps output
stylistically consistent with the human reports.
"""
from __future__ import annotations
from pathlib import Path

from .schema import (
    FrameAnalysis,
    TipoFondale,
    Granulometria,
    CoperturaAlgale,
    TaxaAlgali,
    ElementoAntropico,
    TipoFauna,
)


_FONDALE_LABEL = {
    TipoFondale.SABBIOSO: "Fondale sabbioso",
    TipoFondale.GHIAIOSO: "Fondale ghiaioso",
    TipoFondale.MISTO_SABBIA_GHIAIA: "Fondale sabbioso misto a ghiaia",
    TipoFondale.ROCCIOSO: "Fondale roccioso",
    TipoFondale.ANTROPICO: "Fondale costituito da rocce antropiche",
}

_GRANULO_LABEL = {
    Granulometria.FINE: "di taglia fine",
    Granulometria.MEDIA: "di taglia media",
    Granulometria.GROSSOLANA: "di taglia grossolana",
    Granulometria.CIOTTOLI_PICCOLI: "con ciottoli di piccola taglia",
    Granulometria.CIOTTOLI_MEDI: "con ciottoli di taglia media",
}

_COVER_CLAUSE = {
    CoperturaAlgale.ASSENTE: None,
    CoperturaAlgale.SPORADICA: "presenza sporadica di vegetazione algale",
    CoperturaAlgale.DIFFUSA: "presenza di vegetazione algale",
    CoperturaAlgale.DOMINANTE: "diffusa copertura di vegetazione algale",
}

_ELEMENTO_LABEL = {
    ElementoAntropico.ROCCE_ANTROPICHE: "rocce antropiche",
    ElementoAntropico.RIFIUTO_PLASTICO: "rifiuti antropici di origine plastica",
    ElementoAntropico.RIFIUTO_METALLICO: "rifiuti metallici di origine antropica",
    ElementoAntropico.TUBO_CONDOTTA: "una condotta sottomarina affiorante dal substrato",
    ElementoAntropico.PNEUMATICO: "uno pneumatico",
    ElementoAntropico.RETE_DA_PESCA: "rete da pesca abbandonata",
    ElementoAntropico.PARTI_RELITTO: "parti che potrebbero ricondursi a un relitto",
    ElementoAntropico.TRAVE_LEGNO: "una trave in legno",
}

_FAUNA_LABEL = {
    TipoFauna.STELLA_MARINA: "stella marina",
    TipoFauna.RICCIO_DI_MARE: "riccio di mare",
    TipoFauna.MOLLUSCO_BIVALVE: "mollusco bivalve",
}


def render_caption(a: FrameAnalysis) -> str:
    parts: list[str] = [_FONDALE_LABEL[a.tipo_fondale]]

    if a.granulometria:
        parts.append(_GRANULO_LABEL[a.granulometria])

    rock_clauses = []
    if a.presenza_rocce:
        rock_clauses.append("rocce")
    if a.presenza_ciottoli:
        rock_clauses.append("ciottoli")

    cover = _COVER_CLAUSE[a.copertura_algale]
    if rock_clauses and cover:
        parts.append(
            f"con presenza di {' e '.join(rock_clauses)} ricoperti da vegetazione algale"
        )
    elif rock_clauses:
        parts.append(f"con presenza di {' e '.join(rock_clauses)}")
    elif cover:
        parts.append(cover)

    named_taxa = [t.value for t in a.taxa_algali if t != TaxaAlgali.NON_IDENTIFICATA]
    if named_taxa:
        parts.append(f"tra cui probabilmente {', '.join(named_taxa)}")

    elementi = a.elementi_antropici
    if (
        a.tipo_fondale == TipoFondale.ANTROPICO
        and ElementoAntropico.ROCCE_ANTROPICHE in elementi
    ):
        elementi = [e for e in elementi if e != ElementoAntropico.ROCCE_ANTROPICHE]
    if elementi:
        items = ", ".join(_ELEMENTO_LABEL[e] for e in elementi)
        parts.append(f"presenza di {items}")

    if a.fauna:
        fauna_strs = []
        for f in a.fauna:
            label = _FAUNA_LABEL[f.tipo]
            if f.specie:
                fauna_strs.append(f"{label} ({f.specie})")
            else:
                fauna_strs.append(label)
        parts.append(f"presenza di {', '.join(fauna_strs)}")

    return ", ".join(p for p in parts if p) + "."


def _format_time(t_sec: float) -> str:
    mm = int(t_sec // 60)
    ss = int(t_sec % 60)
    return f"{mm:02d}:{ss:02d}"


def render_markdown_report(
    items: list[tuple[FrameAnalysis, dict]],
    out_path: Path,
    title: str = "Report di ispezione ROV",
) -> None:
    lines: list[str] = [f"# {title}\n"]
    lines.append(f"_Frame rappresentativi: {len(items)}_\n")

    for i, (a, meta) in enumerate(items, 1):
        ts = meta.get("t_sec", 0.0)
        depth = meta.get("depth_m")
        lat = meta.get("lat")
        lon = meta.get("lon")

        depth_s = f", profondità {depth:.1f} m" if depth is not None else ""
        coord_s = (
            f", coord. {lat:.5f}°, {lon:.5f}°"
            if lat is not None and lon is not None
            else ""
        )
        img_rel = meta.get("image_rel", f"frame_{i:03d}.jpg")

        lines.append(f"\n## Figura {i} — t={_format_time(ts)}{depth_s}{coord_s}\n")
        lines.append(f"![Figura {i}]({img_rel})\n")
        lines.append(f"**Descrizione:** {render_caption(a)}\n")
        if a.note:
            lines.append(f"\n_Note:_ {a.note}\n")

    out_path.write_text("\n".join(lines), encoding="utf-8")
