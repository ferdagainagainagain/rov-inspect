"""
Shared VLM prompt and response-parsing logic.

The same Italian system+user prompt is used by both the local
(mlx-vlm) and API (Anthropic) backends so outputs are directly
comparable in ablations.
"""
from __future__ import annotations
import json

from .schema import FrameAnalysis


SYSTEM_PROMPT = """You are an expert marine biologist and environmental technician.
You analyze single frames extracted from underwater inspection videos
recorded by ROVs in Italian seabeds (coastal and port areas).

Respond ONLY with valid JSON, no markdown fences, no preamble. Do not
add any text before or after the JSON, and do not wrap it in markdown
code blocks.

Be conservative in taxonomic identifications: report a genus or species
ONLY when the visible morphological traits reasonably support it; when
in doubt, use 'non_identificata'."""


def user_prompt(schema_json: str) -> str:
    return f"""Analyze this frame and produce a JSON conforming to the following schema:

{schema_json}

Operational guidelines:
- REASONING PROCESS: First, examine the FULL frame in English. What is
  the DOMINANT feature — is it sand, gravel, or rock? Do not default to
  "sabbioso" just because some sand is visible. If rocks or boulders
  occupy more than ~30% of the frame, the dominant category is
  rocce_antropiche or roccioso, NOT sabbioso. Only after determining the
  dominant feature in English, emit the appropriate Italian enum value.

- GENERAL PRINCIPLE: when in doubt, OMIT. It is far better not to report
  an uncertain element than to identify it incorrectly.

- The ROV has a POWER CABLE (tether) that is often visible as a braided
  red, black, or yellow line/rope. This is NOT a tubo_condotta, NOT a
  rete_da_pesca, NOT fauna. It is inspection equipment. NEVER include it
  in elementi_antropici.

- The ROV may project GREEN OR RED LASER POINTS/LINES onto the seabed for
  scale reference. These are inspection equipment, NOT biological
  features, NOT anthropogenic objects, and NOT to be reported in any
  field. Ignore them completely.

- 'tipo_fondale' describes the DOMINANT composition visible in the frame
  (e.g. tipo_fondale: 'sabbioso' for pure sand).

- 'granulometria' applies ONLY to gravel or mixed seabeds, NEVER to pure
  sandy seabeds.

- Do NOT assign granulometria when tipo_fondale is rocce_antropiche or
  roccioso. Those categories describe boulders/large stones, not granules
  or cobbles. Leave granulometria null in those cases.

- 'copertura_algale': observe the color and texture of the rocks. Rocks
  covered with algae show uniform green/brown/dark surface tones; bare
  rocks show grey or light brown tones. This is an IMPORTANT distinction
  for the reports.

- Sandy bottoms with shadow variation, ripples, current-induced texture,
  or color gradients are NOT algal coverage. Algal coverage requires
  visible green/brown/red biological growth attached to a solid surface
  (rocks, cobbles, anthropic stones). On pure sand, copertura_algale
  should almost always be assente.

- Anthropic stones (rocce_antropiche) vs natural rocks: anthropic stones
  are angular, similarly-sized, and often arranged as breakwater
  structures. Natural rocks are irregular and varied in size.

- If the scene is dominated by tightly-packed angular stones of similar
  size (typical of port breakwater construction), the seabed type is
  rocce_antropiche, not roccioso.

- 'taxa_algali': include a genus ONLY if you see distinctive morphological
  traits:
    * Asparagopsis: visible feathery shape, reddish
    * Corallinales: visible pink/calcareous encrustations
    * Peyssonelia: visible overlapping laminae
  In ALL other cases, leave the list empty or use 'non_identificata'.

- 'elementi_antropici': ONLY clearly non-natural objects that are
  unequivocally identifiable. Do NOT include uncertain objects.

- 'fauna': identify an animal ONLY if the shape is unequivocally that of
  an organism (star-shaped body for starfish, spiny sphere for sea
  urchins, bivalve shell). Lines, ropes, elongated shapes are NOT fauna.
  When in doubt, do NOT include fauna.

- 'qualita_visuale' (0-1): lower it drastically for frames showing the
  water surface, bubbles, reflections, or poor seabed visibility.

- 'note' (optional, max 1-2 sentences): relevant details not captured by
  the structured categories. Do NOT repeat what is already in the fields.

Respond ONLY with valid JSON, no markdown fences, no preamble."""


def schema_json_str() -> str:
    return json.dumps(
        FrameAnalysis.model_json_schema(),
        indent=2,
        ensure_ascii=False,
    )


def strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text[3:]
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
        if text.endswith("```"):
            text = text[:-3].strip()
    return text


def extract_first_json_object(text: str) -> str:
    """Extract the first balanced {...} block from text.

    Local models occasionally append commentary despite instructions;
    this is a forgiving fallback. Returns the original text if no
    balanced object is found (let Pydantic raise a clear error).
    """
    start = text.find("{")
    if start == -1:
        return text
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
    return text


def parse_response(text: str) -> FrameAnalysis:
    text = strip_code_fences(text)
    text = extract_first_json_object(text)
    return FrameAnalysis.model_validate_json(text)
