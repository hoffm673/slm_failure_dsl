"""
Failure-mode detectors.

Each detector takes (step_output, step_input_context, raw_text) and returns
True if the failure mode is present. These are the mechanical witnesses that
turn an LLM output into a classified failure.

IMPORTANT: detectors are intentionally noisy/imperfect. The eval is honest
about detector precision/recall -- spot-check a sample by hand and report it.
This is standard in LLM-eval papers.

After reading the taxonomy paper, you'll want to:
  - Add / remove detectors to match the failure modes you commit to.
  - Possibly add a "confidence" score per detector and threshold it.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable

from .core import FailureMode


# ---------- SCHEMA_VIOLATION ----------

def detect_schema_violation(
    output: Any, schema: dict | None, raw_text: str | None
) -> bool:
    """
    If the step declares a JSON-schema-like dict, check that `output` matches.
    A minimal implementation -- swap in `jsonschema` library for real use.
    """
    if schema is None:
        return False
    return not _matches_schema(output, schema)


def _matches_schema(value: Any, schema: dict) -> bool:
    expected = schema.get("type")
    if expected == "object":
        if not isinstance(value, dict):
            return False
        for field_name, field_schema in schema.get("properties", {}).items():
            if field_name not in value:
                if field_name in schema.get("required", []):
                    return False
                continue
            if not _matches_schema(value[field_name], field_schema):
                return False
        return True
    if expected == "array":
        if not isinstance(value, list):
            return False
        item_schema = schema.get("items")
        if item_schema:
            return all(_matches_schema(x, item_schema) for x in value)
        return True
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    return True  # unknown / permissive


# ---------- HALLUCINATED_ENTITY ----------

def detect_hallucinated_entity(
    output_text: str,
    source_context: str,
    entity_extractor: Callable[[str], set[str]] | None = None,
) -> bool:
    """
    Cheap heuristic: extract proper-noun-ish tokens from output, check that
    each appears in the source context. Real implementation should use an NER
    model. For SLM eval, even this crude version finds clear cases.
    """
    if entity_extractor is None:
        entity_extractor = _proper_noun_tokens
    output_entities = entity_extractor(output_text)
    source_entities = entity_extractor(source_context)
    # normalize
    output_entities = {e.lower() for e in output_entities}
    source_entities = {e.lower() for e in source_entities}
    unsupported = output_entities - source_entities
    # filter common false positives (months, days, etc.)
    unsupported -= _COMMON_PROPER_NOUNS
    return len(unsupported) > 0


def _proper_noun_tokens(text: str) -> set[str]:
    # crude: any capitalized token of length >= 3 that isn't at sentence start
    # better: spaCy NER; left as TODO
    return set(re.findall(r"(?<!^)(?<![.!?]\s)\b[A-Z][a-zA-Z]{2,}\b", text))


_COMMON_PROPER_NOUNS = {
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
    "i", "english",
}


# ---------- LATENT_INCONSISTENCY ----------
# detected at the harness level by running the same input N times and comparing
# outputs across runs. See harness.py.


# ---------- INSTRUCTION_NEGLECT ----------

def detect_instruction_neglect(
    output_text: str,
    required_phrases: list[str] | None = None,
    forbidden_phrases: list[str] | None = None,
    max_length: int | None = None,
) -> bool:
    """
    Declarative: caller specifies what the prompt told the model to do/not-do,
    detector checks compliance. Crude but effective for explicit instructions
    like "respond in JSON" or "do not use the word X".
    """
    if required_phrases:
        for p in required_phrases:
            if p.lower() not in output_text.lower():
                return True
    if forbidden_phrases:
        for p in forbidden_phrases:
            if p.lower() in output_text.lower():
                return True
    if max_length is not None and len(output_text) > max_length:
        return True
    return False


# ---------- REFUSAL ----------

_REFUSAL_PATTERNS = [
    r"\bI can('?| no)t\b",
    r"\bI(?:'m| am) (?:unable|not able)\b",
    r"\bas an AI\b",
    r"\bI('?m| am) sorry,? but\b",
    r"\bI(?:'m| am) afraid I can",
]


def detect_refusal(output_text: str) -> bool:
    return any(re.search(p, output_text, re.IGNORECASE) for p in _REFUSAL_PATTERNS)


# ---------- TOOL_INVOCATION_ERROR ----------

def detect_tool_invocation_error(
    parsed_call: dict | None,
    valid_tool_names: set[str],
    tool_schemas: dict[str, dict],
) -> bool:
    """
    parsed_call should be {"name": "...", "args": {...}} or None if parsing failed.
    """
    if parsed_call is None:
        return True
    name = parsed_call.get("name")
    if name not in valid_tool_names:
        return True
    args = parsed_call.get("args", {})
    schema = tool_schemas.get(name)
    if schema is not None and not _matches_schema(args, schema):
        return True
    return False


# ---------- registry ----------

DETECTORS = {
    FailureMode.SCHEMA_VIOLATION: detect_schema_violation,
    FailureMode.HALLUCINATED_ENTITY: detect_hallucinated_entity,
    FailureMode.INSTRUCTION_NEGLECT: detect_instruction_neglect,
    FailureMode.REFUSAL: detect_refusal,
    FailureMode.TOOL_INVOCATION_ERROR: detect_tool_invocation_error,
    # LATENT_INCONSISTENCY: harness-level
    # CONTEXT_BOUNDARY_DEGRADATION: harness-level (compare base vs. padded)
    # REASONING_DRIFT: requires reasoning trace, defer
}
