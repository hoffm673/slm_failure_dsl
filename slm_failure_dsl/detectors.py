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

def detect_non_entity_in_entities(entities: list, source: str) -> bool:
    """
    A capitalized-only heuristic isn't enough. Flag entities that are common
    nouns (lowercase in the source) being treated as entities.
    """
    if not isinstance(entities, list):
        return False
    source_lower = source.lower()
    for e in entities:
        if not isinstance(e, str):
            continue
        # if the entity appears in the source ONLY in lowercase form,
        # it's probably a common noun being miscategorized
        if e.lower() in source_lower:
            # check if it appears capitalized in source
            if e not in source and e.capitalize() not in source:
                # appears only in lowercase form -> probably not a proper noun
                return True
    return False
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

_DAYS_MONTHS_STOPS = {
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
    "today", "tomorrow", "yesterday",
}
 
# Titles look like proper nouns but aren't standalone entities.
_TITLES = {
    "mr", "mrs", "ms", "miss", "dr", "prof", "professor", "sir", "lord",
    "lady", "rev", "reverend", "hon", "honorable", "honourable",
    "sen", "senator", "rep", "representative", "gov", "governor",
    "pres", "president",
}
 
# Sentence-initial words that aren't entities even when capitalized.
_SENTENCE_INITIAL_STOPWORDS = {
    # function words
    "the", "a", "an", "this", "that", "these", "those",
    "it", "they", "we", "i", "he", "she", "you",
    "but", "and", "or", "so", "if", "when", "while", "after", "before",
    "however", "moreover", "meanwhile", "still", "yet", "also",
    "last", "next", "first", "second", "third", "fourth",
    "there", "here", "now", "then",
    "both", "either", "neither", "some", "any", "all",
    "during", "throughout",
    # common nouns often appearing sentence-initial in news/business text;
    # without this list, rule (c) would over-flag (e.g. "Investors responded...")
    "ceo", "cto", "cfo", "coo", "founder", "executive", "executives",
    "investors", "shareholders", "regulators", "officials", "analysts",
    "customers", "users", "employees", "workers", "researchers",
    "scientists", "doctors", "lawyers", "experts", "critics",
    "company", "companies", "team", "teams", "group", "groups",
    "according", "reports", "sources", "results", "data",
}
 
_PROPER_NOUN_RE = re.compile(r"\b[A-Z][a-zA-Z][a-zA-Z0-9.\-]*\b")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_TOKEN_RE = re.compile(r"\b[A-Za-z][A-Za-z\-]*\b")
 
 
# ---------------------------------------------------------------------------
# 1. ENTITY_OMISSION
# ---------------------------------------------------------------------------
# A salient entity from the source is missing from the output's entity list.
#
# Salience rule, tuned for low false positives:
#   A token is salient iff (after lowercasing) it is NOT in any stop list
#   AND either
#     (a) it appears capitalized at least once mid-sentence (strong proof
#         it's a proper noun, not just sentence-initial capitalization), OR
#     (b) it appears capitalized in at least 2 distinct sentences
#         (recurrence is also strong evidence)
#
# Then we check each salient entity appears (substring, case-insensitive)
# in the flattened output entity list.
 
def _extract_salient_entities(source: str) -> set[str]:
    """
    Return the set of source tokens we'll require to appear in the output.
    Deliberately conservative.
    """
    # tok_lower -> {"mid": int, "sentences": set of sentence indices seen in}
    stats: dict[str, dict] = {}
    sentences = _SENTENCE_SPLIT_RE.split(source.strip())
 
    for s_idx, sentence in enumerate(sentences):
        for m in _PROPER_NOUN_RE.finditer(sentence):
            lower = m.group().lower()
            entry = stats.setdefault(lower, {"mid": 0, "sentences": set()})
            entry["sentences"].add(s_idx)
            # mid-sentence = not at the start of the sentence
            preceding = sentence[:m.start()]
            if preceding.strip():
                entry["mid"] += 1
 
    salient = set()
    for lower, info in stats.items():
        if lower in _DAYS_MONTHS_STOPS:
            continue
        if lower in _SENTENCE_INITIAL_STOPWORDS:
            continue
        if lower in _TITLES:
            continue
        # Rule (a): mid-sentence capitalization — strong evidence
        # Rule (b): appears capitalized across 2+ sentences — also strong
        # Rule (c): sentence-initial only, but length >= 3 and not in
        #   stop lists — weaker but necessary to catch lead-subject entities
        #   like "Apple released..." where the article opens with the main entity
        if info["mid"] >= 1 or len(info["sentences"]) >= 2 or len(lower) >= 3:
            salient.add(lower)
    return salient
 
 
def _entities_to_strings(entities: Any) -> list[str]:
    """
    Flatten the output `entities` field to a list of strings, accepting
    either ['Apple', 'Tim Cook'] or [{'name': 'Apple', 'role': 'company'}, ...].
    """
    if not isinstance(entities, list):
        return []
    out = []
    for e in entities:
        if isinstance(e, str):
            out.append(e)
        elif isinstance(e, dict):
            name = e.get("name") or e.get("entity") or e.get("text")
            if isinstance(name, str):
                out.append(name)
    return out
 
 
def detect_entity_omission(
    output: Any,
    source_context: str,
    entities_field: str = "entities",
) -> bool:
    """
    Fires if any salient entity from the source is missing from the output.
 
    `output` may be the parsed JSON dict (we look for output[entities_field])
    or a list of entities directly.
    """
    if isinstance(output, dict):
        entities = output.get(entities_field, [])
    else:
        entities = output
 
    output_strings = _entities_to_strings(entities)
    output_haystack = " ".join(output_strings).lower()
 
    salient = _extract_salient_entities(source_context)
    if not salient:
        return False
 
    for required in salient:
        if required not in output_haystack:
            return True
    return False
 
 
# ---------------------------------------------------------------------------
# 2. COMMON_NOUN_AS_ENTITY
# ---------------------------------------------------------------------------
# Fires when an entity in the output is a single-token term that appears
# strictly lowercase mid-sentence in the source AND never appears
# capitalized mid-sentence.
#
# This catches "restaurant"/"chef" without flagging "Microsoft" (which
# appears capitalized — sentence-initial or not).
#
# Multi-word entities are not flagged here — they're handled by
# hallucinated_entity if hallucinated.
 
def _appears_strictly_lowercase_midsentence(source: str, term: str) -> bool:
    """Does `term` appear strictly lowercase in a non-sentence-initial position?"""
    term_lower = term.lower()
    sentences = _SENTENCE_SPLIT_RE.split(source.strip())
    for sentence in sentences:
        for i, m in enumerate(_TOKEN_RE.finditer(sentence)):
            if m.group().lower() != term_lower:
                continue
            if i == 0:  # sentence-initial
                continue
            if m.group() == m.group().lower():
                return True
    return False
 
 
def _appears_capitalized_midsentence(source: str, term: str) -> bool:
    """Does `term` appear capitalized in a non-sentence-initial position?"""
    term_lower = term.lower()
    sentences = _SENTENCE_SPLIT_RE.split(source.strip())
    for sentence in sentences:
        for i, m in enumerate(_TOKEN_RE.finditer(sentence)):
            if m.group().lower() != term_lower:
                continue
            if i == 0:
                continue
            if m.group()[0].isupper():
                return True
    return False
 
 
def detect_common_noun_as_entity(
    output: Any,
    source_context: str,
    entities_field: str = "entities",
) -> bool:
    """
    Fires if any single-token output entity appears strictly lowercase
    mid-sentence in the source and never appears capitalized mid-sentence.
    """
    if isinstance(output, dict):
        entities = output.get(entities_field, [])
    else:
        entities = output
 
    for ent_str in _entities_to_strings(entities):
        ent_str = ent_str.strip()
        if not ent_str or " " in ent_str:
            continue
        if (_appears_strictly_lowercase_midsentence(source_context, ent_str)
                and not _appears_capitalized_midsentence(source_context, ent_str)):
            return True
    return False
 
 
# ---------------------------------------------------------------------------
# 3. SENTIMENT_MISMATCH
# ---------------------------------------------------------------------------
# Fires when the model's declared sentiment label contradicts a lexicon-based
# signal computed from the source passage.
#
# Only fires when the source has a clear dominant polarity (|pos - neg| > 3)
# AND the model returns the opposite label. "neutral" is never penalised —
# it's always a defensible hedge.

_POSITIVE_WORDS = {
    "good", "great", "excellent", "positive", "success", "successful",
    "improved", "improvement", "growth", "strong", "innovative", "remarkable",
    "winning", "progress", "thriving", "better", "best", "pleased",
    "satisfied", "approved", "landmark", "record", "surpassed", "optimistic",
    "promising", "effective", "outstanding", "praised", "celebrated",
    "confirmed", "launched", "achieved", "beat", "exceeded",
}

_NEGATIVE_WORDS = {
    "bad", "poor", "failed", "failure", "decline", "declining", "drop",
    "fell", "fall", "loss", "concern", "worried", "problem", "issue",
    "crisis", "slow", "rejected", "charged", "arrested", "collapsed",
    "disappointing", "inadequate", "insufficient", "rude", "lost",
    "cut", "cuts", "lawsuit", "penalty", "fine", "deficit", "shortage",
    "destroyed", "damaged", "collapsed", "devastated", "emergency",
    "collapsed", "broke", "convicted", "sentenced", "condemned",
}


def detect_sentiment_mismatch(model_sentiment: str, source_context: str) -> bool:
    """
    Fires when the model labels sentiment as the clear opposite of what a
    simple word-count over the source suggests.
    """
    words = re.findall(r"\b[a-z]+\b", source_context.lower())
    word_set = set(words)
    pos = len(word_set & _POSITIVE_WORDS)
    neg = len(word_set & _NEGATIVE_WORDS)
    if abs(pos - neg) <= 3:
        return False  # ambiguous — don't flag
    dominant = "positive" if pos > neg else "negative"
    opposite = "negative" if dominant == "positive" else "positive"
    return model_sentiment == opposite


# ---------------------------------------------------------------------------
# 4. OUTPUT_TRUNCATION
# ---------------------------------------------------------------------------
# Fires when the raw model output appears to have been cut off mid-generation.
# A well-formed response ends with sentence-terminal or closing punctuation.
# A truncated one ends mid-word, mid-number, or with a comma/colon.

_TERMINAL_CHARS = frozenset('.!?"\']})')


def detect_output_truncation(raw_text: str | None) -> bool:
    """
    Fires when raw_text ends without terminal punctuation, suggesting the
    model hit max_new_tokens before finishing. Distinct root cause from
    schema_violation (formatting intent vs. capacity).
    """
    if not raw_text:
        return False
    stripped = raw_text.rstrip()
    if len(stripped) < 40:
        return False  # too short to distinguish truncation from refusal
    return stripped[-1] not in _TERMINAL_CHARS


# ---------------------------------------------------------------------------
# REGISTRY ADDITIONS — drop these into detectors.py / core.py
# ---------- registry ----------

DETECTORS = {
    FailureMode.SCHEMA_VIOLATION: detect_schema_violation,
    FailureMode.HALLUCINATED_ENTITY: detect_hallucinated_entity,
    FailureMode.INSTRUCTION_NEGLECT: detect_instruction_neglect,
    FailureMode.REFUSAL: detect_refusal,
    FailureMode.TOOL_INVOCATION_ERROR: detect_tool_invocation_error,
    FailureMode.ENTITY_OMISSION:       detect_entity_omission,
    FailureMode.COMMON_NOUN_AS_ENTITY: detect_common_noun_as_entity,
    FailureMode.SENTIMENT_MISMATCH:    detect_sentiment_mismatch,
    FailureMode.OUTPUT_TRUNCATION:     detect_output_truncation,

    # LATENT_INCONSISTENCY: harness-level
    # CONTEXT_BOUNDARY_DEGRADATION: harness-level (compare base vs. padded)
    # REASONING_DRIFT: requires reasoning trace, defer
}
