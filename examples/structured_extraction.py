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
    detect_entity_omission,
    detect_common_noun_as_entity,
    detect_sentiment_mismatch,
    detect_output_truncation,
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
            FailureMode.INSTRUCTION_NEGLECT,
            FailureMode.TOOL_INVOCATION_ERROR,
            FailureMode.ENTITY_OMISSION,
            FailureMode.COMMON_NOUN_AS_ENTITY,
            FailureMode.SENTIMENT_MISMATCH,
            FailureMode.OUTPUT_TRUNCATION,
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
        if detect_output_truncation(raw):
            fired.append(FailureMode.OUTPUT_TRUNCATION)

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

        # semantic detectors: entity omission, common-noun-as-entity, sentiment
        if parsed and isinstance(parsed, dict):
            if detect_entity_omission(parsed, passage):
                fired.append(FailureMode.ENTITY_OMISSION)
            if detect_common_noun_as_entity(parsed, passage):
                fired.append(FailureMode.COMMON_NOUN_AS_ENTITY)
            sentiment = parsed.get("sentiment", "")
            if isinstance(sentiment, str) and detect_sentiment_mismatch(sentiment, passage):
                fired.append(FailureMode.SENTIMENT_MISMATCH)

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
    # 1. ambiguous entities (is "Apple" the company or fruit?)
    {"passage": "Apple released its quarterly results today. The stock fell 3% as investors worried about iPhone demand in China. Tim Cook said the company remains optimistic about services revenue."},

    # 2. multiple entities, some only referenced by pronoun
    {"passage": "Last Tuesday, Dr. Sarah Chen and her colleague presented findings at MIT. While she focused on the theoretical framework, he handled the empirical results. The audience, including Professor Martinez from Stanford, raised concerns about reproducibility."},

    # 3. sentiment that flips mid-passage (stresses sentiment_mismatch + common_noun_as_entity)
    {"passage": "The new restaurant has incredible food and the chef trained at top kitchens in Paris. However, the service was painfully slow, our reservation was lost, and the manager was rude when we complained."},

    # 4. nested information, requires inference
    {"passage": "The acquisition fell through after regulators in three jurisdictions raised antitrust concerns. Shareholders of both Beta Corp and Acme had previously approved the merger at $4.2B."},

    # 5. many entities across countries and organisations
    {"passage": "Climate negotiations in Geneva concluded with a non-binding agreement. The delegation from Brazil, led by Minister Silva, pushed for stronger forest protections. France, Germany, and Japan announced new funding pledges totaling $12 billion. Critics including Greta Thunberg called the outcome insufficient. The next meeting is scheduled for December in Nairobi."},

    # 6. technical jargon (stresses hallucinated_entity with architecture names)
    {"passage": "The PR introduces a CUDA kernel optimization for the attention mechanism that reduces memory bandwidth by approximately 40% on Ampere GPUs. The author, working at Anthropic, notes that the optimization may not help on Hopper or Blackwell architectures."},

    # 7. subtle factual claim with chained proper nouns
    {"passage": "Marie Curie was the first woman to win a Nobel Prize, and the only person ever to win Nobels in two different sciences. Her daughter Irene also became a Nobel laureate."},

    # 8. legal framing that invites refusal-shaped output even though benign
    {"passage": "The defendant was charged with embezzlement after auditors found irregularities. The trial begins next month in district court."},

    # 9. medical trial — many proper nouns, domain nouns (trial, sample, tumor) that
    #    small models misclassify as entities; clearly positive sentiment
    {"passage": "Researchers at the Mayo Clinic and Stanford Medical Center reported that the immunotherapy drug Keytruda reduced disease progression in non-small-cell lung cancer by 41 percent compared to standard chemotherapy. Lead investigator Dr. Rachel Kim noted the trial enrolled 340 patients across twelve sites. Merck, which manufactures Keytruda, saw its stock rise 5 percent following the announcement."},

    # 10. sports — mixed sentiment, many proper nouns, common nouns as false entities
    {"passage": "France defeated England 3-2 in the UEFA Nations League final in Munich, with Kylian Mbappe scoring twice in the second half. England manager Gareth Southgate called the result heartbreaking but praised his team's effort. The match drew a television audience of 28 million viewers across the United Kingdom."},

    # 11. historical — neutral sentiment, many names, pronoun coreference across sentences
    {"passage": "The Cuban Missile Crisis of October 1962 brought the United States and the Soviet Union to the brink of nuclear war. President Kennedy and Premier Khrushchev negotiated through back channels, with Soviet ambassador Anatoly Dobrynin serving as a key intermediary. The standoff ended after thirteen days when Soviet ships turned back and the missiles were dismantled."},

    # 12. financial earnings — mixed (beats overall, one division declining)
    {"passage": "Goldman Sachs reported second-quarter net income of 3.4 billion dollars, exceeding analyst forecasts by 12 percent. Chief Executive David Solomon credited strong performance in the investment banking division. However, the consumer banking unit Marcus posted its third consecutive quarterly loss, and the company announced plans to scale back its retail ambitions."},

    # 13. regulatory/antitrust — negative, entities across legal and tech domains
    {"passage": "The Federal Trade Commission filed an antitrust lawsuit against Meta, alleging that the acquisitions of Instagram and WhatsApp were designed to suppress competition. Meta's legal team argued both purchases received regulatory approval at the time and that consumers benefit from integration across platforms. The case is scheduled for trial in the Southern District of New York."},

    # 14. environmental disaster — clearly negative, common nouns (crew, zone, fire) as
    #     false entity candidates
    {"passage": "A wildfire in the Mendocino region of northern California destroyed more than 60,000 acres over five days before crews from CalFire and the National Forest Service achieved full containment. The town of Ukiah was placed under evacuation orders and air quality in the surrounding zone reached hazardous levels. Governor Newsom declared a state of emergency and requested federal support from FEMA."},

    # 15. higher education policy — institutions + officials + common nouns (cycle, merit,
    #     giving) as false entity candidates
    {"passage": "Harvard University announced it would eliminate legacy admissions preferences starting in the 2025 application cycle. President Alan Garber said the decision reflects a commitment to meritocratic selection and follows a review ordered after the Supreme Court's ruling in Students for Fair Admissions v. Harvard. Alumni advocacy groups warned the change could significantly reduce annual giving."},

    # 16. international diplomacy — negative, many countries + officials + pronouns
    {"passage": "Negotiations between Israeli and Palestinian delegations in Cairo collapsed after four days without an agreement on a proposed ceasefire framework. Egyptian Foreign Minister Sameh Shoukry expressed deep disappointment and urged both parties to return to the table. United Nations Special Coordinator Tor Wennesland warned that continued fighting would worsen the humanitarian situation in Gaza."},

    # 17. real estate / economics — neutral-to-negative, financial common nouns everywhere
    {"passage": "Manhattan office vacancy rates climbed to 23 percent in the third quarter, the highest level recorded since 1995, according to a market report from Cushman and Wakefield. The Financial District and Hudson Yards submarkets both posted double-digit vacancy increases. Landlords including SL Green Realty and Brookfield Asset Management announced plans to convert underutilised towers to residential use."},

    # 18. agriculture — neutral, domain nouns (futures, yield, bushel) that small models
    #     frequently treat as entities
    {"passage": "A prolonged drought across Iowa, Illinois, and Indiana reduced corn yields by an estimated 19 percent compared to the five-year average, according to a preliminary assessment from the United States Department of Agriculture. Commodity prices rose sharply, with December corn futures on the Chicago Mercantile Exchange settling at a two-year high. Farm bureau representatives from the affected states requested emergency assistance from the USDA."},

    # 19. space / aerospace — clearly positive, dense with proper nouns and acronyms
    {"passage": "NASA and SpaceX confirmed the successful separation of the Orion capsule from the Space Launch System rocket eight minutes after launch from Kennedy Space Center in Florida. Commander Anne McClain and mission specialist Victor Glover are expected to reach lunar orbit within three days. Flight directors at the Johnson Space Center in Houston described telemetry as nominal across all systems."},

    # 20. criminal justice — clearly negative, chained proper nouns + legal common nouns
    {"passage": "A federal jury in Chicago convicted former Illinois Governor Rod Blagojevich on all fourteen counts of corruption, including attempted extortion and wire fraud related to his effort to sell a vacant United States Senate seat. Judge James Zagel sentenced him to fourteen years in federal prison, calling the conduct a profound abuse of the public trust. Blagojevich's attorney, Sheldon Sorosky, announced plans to appeal the conviction."},
]
