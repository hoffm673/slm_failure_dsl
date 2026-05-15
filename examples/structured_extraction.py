"""
Example pipeline: structured extraction from a passage.

Stages:
  1. extract  : LLM call -> JSON with {summary, entities, sentiment}
  2. validate : parse + schema-check the JSON
  3. enrich   : add a `confidence` field based on heuristics

This is the simplest pipeline and the best one to debug the harness with.
"""

from __future__ import annotations

import json
import re

from slm_failure_dsl import (
    FailureMode,
    Pipeline,
    StepResult,
    step,
)
from slm_failure_dsl.detectors import (
    detect_hallucinated_entity,
    detect_refusal,
    detect_schema_violation,
)


EXTRACT_SCHEMA = {
    "type": "object",
    "required": ["summary", "entities", "sentiment"],
    "properties": {
        "summary": {"type": "string"},
        "entities": {"type": "array", "items": {"type": "string"}},
        "sentiment": {"type": "string"},
    },
}


def build_pipeline(model) -> Pipeline:
    """`model` is a ModelBackend (StubModel or TransformersModel)."""

    @step(
        name="extract",
        may_fail=[
            FailureMode.SCHEMA_VIOLATION,
            FailureMode.REFUSAL,
            FailureMode.HALLUCINATED_ENTITY,
        ],
    )
    def extract(ctx: dict) -> StepResult:
        """Call the LLM to extract structured info from `ctx['passage']`."""
        passage = ctx["passage"]
        prompt = (
            "Extract structured information from the following passage. "
            "Respond with ONLY a JSON object with keys: "
            '"summary" (one-sentence string), '
            '"entities" (list of names from the passage), '
            '"sentiment" (one of: positive, negative, neutral).\n\n'
            f"Passage:\n{passage}"
        )
        raw = model.generate(prompt, max_new_tokens=200, temperature=0.3)

        # try to parse JSON out of the response
        parsed = _try_parse_json(raw)
        fired = []
        if detect_refusal(raw):
            fired.append(FailureMode.REFUSAL)
            return StepResult(output=None, fired_failures=fired, raw_text=raw)

        if detect_schema_violation(parsed, EXTRACT_SCHEMA, raw):
            fired.append(FailureMode.SCHEMA_VIOLATION)

        # only attempt hallucination check if entities are well-typed strings
        if (
            parsed
            and isinstance(parsed, dict)
            and isinstance(parsed.get("entities"), list)
        ):
            entity_strs = [e for e in parsed["entities"] if isinstance(e, str)]
            if entity_strs:
                entity_text = " ".join(entity_strs)
                if detect_hallucinated_entity(entity_text, passage):
                    fired.append(FailureMode.HALLUCINATED_ENTITY)

        return StepResult(output=parsed, fired_failures=fired, raw_text=raw)

    @step(
        name="validate",
        depends_on=["extract"],
        may_fail=[FailureMode.SCHEMA_VIOLATION],
        recovers=[FailureMode.SCHEMA_VIOLATION],  # we coerce or default
    )
    def validate(ctx: dict) -> StepResult:
        """Coerce missing fields to safe defaults. Recovers SCHEMA_VIOLATION."""
        raw = ctx.get("extract") or {}
        if not isinstance(raw, dict):
            raw = {}
        # coerce entities to a list of strings no matter what shape we got
        raw_entities = raw.get("entities", [])
        if isinstance(raw_entities, list):
            entities = []
            for e in raw_entities:
                if isinstance(e, str):
                    entities.append(e)
                elif isinstance(e, dict):
                    # common SLM mistake: {"name": "Alice"} or {"entity": "Alice"}
                    for v in e.values():
                        if isinstance(v, str):
                            entities.append(v)
                            break
        else:
            entities = []

        coerced = {
            "summary": raw.get("summary", "") if isinstance(raw.get("summary"), str) else "",
            "entities": entities,
            "sentiment": raw.get("sentiment", "neutral"),
        }
        if coerced["sentiment"] not in ("positive", "negative", "neutral"):
            coerced["sentiment"] = "neutral"
        return StepResult(output=coerced)

    @step(name="enrich", depends_on=["validate"])
    def enrich(ctx: dict) -> dict:
        v = ctx["validate"]
        confidence = 0.5
        if v["summary"] and v["entities"]:
            confidence = 0.9
        elif not v["summary"]:
            confidence = 0.2
        return {**v, "confidence": confidence}

    p = Pipeline("structured_extraction").add(extract, validate, enrich)
    return p


def _try_parse_json(text: str):
    """Best-effort JSON extraction from an LLM response."""
    if not isinstance(text, str):
        return None
    # try whole string
    try:
        return json.loads(text)
    except Exception:
        pass
    # try first {...} block
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


SEED_INPUTS = [
    # ambiguous entities (is "Apple" the company or fruit?)
    {"passage": "Apple released its quarterly results today. The stock fell 3% as investors worried about iPhone demand in China. Tim Cook said the company remains optimistic about services revenue."},

    # multiple entities, some only referenced by pronoun
    {"passage": "Last Tuesday, Dr. Sarah Chen and her colleague presented findings at MIT. While she focused on the theoretical framework, he handled the empirical results. The audience, including Professor Martinez from Stanford, raised concerns about reproducibility."},

    # sentiment that flips mid-passage
    {"passage": "The new restaurant has incredible food and the chef trained at top kitchens in Paris. However, the service was painfully slow, our reservation was lost, and the manager was rude when we complained."},

    # nested information, requires inference
    {"passage": "The acquisition fell through after regulators in three jurisdictions raised antitrust concerns. Shareholders of both Beta Corp and Acme had previously approved the merger at $4.2B."},

    # long passage with distractor entities
    {"passage": "Climate negotiations in Geneva concluded with a non-binding agreement. The delegation from Brazil, led by Minister Silva, pushed for stronger forest protections. France, Germany, and Japan announced new funding pledges totaling $12 billion. Critics including Greta Thunberg called the outcome insufficient. The next meeting is scheduled for December in Nairobi."},

    # technical content with jargon
    {"passage": "The PR introduces a CUDA kernel optimization for the attention mechanism that reduces memory bandwidth by approximately 40% on Ampere GPUs. The author, working at Anthropic, notes that the optimization may not help on Hopper or Blackwell architectures."},

    # subtle factual claim
    {"passage": "Marie Curie was the first woman to win a Nobel Prize, and the only person ever to win Nobels in two different sciences. Her daughter Irene also became a Nobel laureate."},

    # input that invites refusal-shaped output even though it's benign
    {"passage": "The defendant was charged with embezzlement after auditors found irregularities. The trial begins next month in district court."},
]
