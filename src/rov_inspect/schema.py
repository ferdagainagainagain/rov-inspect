"""
Pydantic schema for ROV inspection frame analysis.

Vocabulary derived from the Rov_Immagine.docx ground truth
(Messina dataset, 11 videos, 93 annotated figures).

All categorical values mirror the language used by the human
expert so that VLM outputs and ground-truth captions are
directly comparable. When extending, prefer adding new enum
values (matching GT spelling) over freeform strings.
"""
from __future__ import annotations
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class TipoFondale(str, Enum):
    SABBIOSO = "sabbioso"
    GHIAIOSO = "ghiaioso"
    MISTO_SABBIA_GHIAIA = "sabbioso_misto_a_ghiaia"
    ROCCIOSO = "roccioso"
    ANTROPICO = "rocce_antropiche"


class Granulometria(str, Enum):
    FINE = "fine"
    MEDIA = "media"
    GROSSOLANA = "grossolana"
    CIOTTOLI_PICCOLI = "ciottoli_piccoli"
    CIOTTOLI_MEDI = "ciottoli_medi"


class CoperturaAlgale(str, Enum):
    ASSENTE = "assente"
    SPORADICA = "sporadica"
    DIFFUSA = "diffusa"
    DOMINANTE = "dominante"


class TaxaAlgali(str, Enum):
    ASPARAGOPSIS = "Asparagopsis"
    JANIA = "Jania"
    PEYSSONELIA = "Peyssonelia"
    CORALLINALES = "Corallinales"
    LAMINARIALES = "Laminariales"
    DETRITO_ALGALE = "detrito_algale"
    NON_IDENTIFICATA = "non_identificata"


class ElementoAntropico(str, Enum):
    ROCCE_ANTROPICHE = "rocce_antropiche"
    RIFIUTO_PLASTICO = "rifiuto_plastico"
    RIFIUTO_METALLICO = "rifiuto_metallico"
    TUBO_CONDOTTA = "tubo_condotta"
    PNEUMATICO = "pneumatico"
    RETE_DA_PESCA = "rete_da_pesca"
    PARTI_RELITTO = "parti_di_relitto"
    TRAVE_LEGNO = "trave_in_legno"


class TipoFauna(str, Enum):
    STELLA_MARINA = "stella_marina"
    RICCIO_DI_MARE = "riccio_di_mare"
    MOLLUSCO_BIVALVE = "mollusco_bivalve"


class FaunaItem(BaseModel):
    tipo: TipoFauna
    specie: Optional[str] = Field(
        None,
        description=(
            "Specie tassonomica se identificabile, es. "
            "'Ophidiaster ophidianus', 'Marthasterias glacialis', "
            "'Arbacia lixula', 'Pinna'."
        ),
    )


class FrameAnalysis(BaseModel):
    """Structured analysis of a single frame, mirroring GT vocabulary."""

    tipo_fondale: TipoFondale
    granulometria: Optional[Granulometria] = None
    presenza_rocce: bool = False
    presenza_ciottoli: bool = False

    copertura_algale: CoperturaAlgale = CoperturaAlgale.ASSENTE
    taxa_algali: list[TaxaAlgali] = Field(default_factory=list)

    elementi_antropici: list[ElementoAntropico] = Field(default_factory=list)
    fauna: list[FaunaItem] = Field(default_factory=list)

    qualita_visuale: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Idoneità del frame come immagine rappresentativa "
            "per il report (0-1)."
        ),
    )
    note: Optional[str] = Field(
        None,
        description="Note brevi (1-2 frasi) su elementi non catturati dalle categorie.",
    )
