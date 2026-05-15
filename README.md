# slm_failure_dsl

An effect-typed DSL + property-based audit harness for small-LLM pipelines.

**Research question:** Do small language models (<3B params) exhibit different
failure-mode distributions than larger models when used in multi-step pipelines,
and what does that imply for pipeline design?

**Artifact:** A Python-embedded DSL where pipeline steps declare expected
failure modes (drawn from Vinay 2025, arXiv:2511.19933), a static analysis pass
that tracks effect propagation through the DAG, and a perturbation-driven
audit harness that measures which failure modes actually fire per (pipeline,
model) combination.

## Layout

```
slm_failure_dsl/
  slm_failure_dsl/        # the library
    core.py               # Pipeline, Step, FailureMode, decorators
    compiler.py           # topo sort + static effect-leakage analysis
    runtime.py            # pipeline execution + RunTrace
    detectors.py          # mechanical failure-mode detectors
    perturbations.py      # input perturbation strategies
    harness.py            # the audit loop -> AuditReport
    backends.py           # StubModel + TransformersModel
  examples/
    structured_extraction.py   # pipeline #1: extract -> validate -> enrich
  experiments/
    smoke_test.py         # runs end-to-end with StubModel, no GPU needed
    run_experiment.py     # CLI: run audit, dump JSON results
```

## Quick start

```bash
cd slm_failure_dsl
python -m experiments.smoke_test
```

This runs with the deterministic stub model and prints a fire-rate table. No
ML deps required.

To run with a real model (needs `transformers`, `torch`, GPU):

```bash
pip install transformers torch accelerate
python -m experiments.run_experiment --model_id Qwen/Qwen2.5-0.5B-Instruct
python -m experiments.run_experiment --model_id Qwen/Qwen2.5-3B-Instruct
```

Results land in `results/` as JSON.

## What's done in this skeleton

- DSL: `@step` decorator with effect signatures, `Pipeline` builder
- Static analysis: topo sort, cycle detection, forward effect propagation,
  dead-recovery detection, unhandled-at-boundary computation
- Runtime: pipeline execution with RunTrace
- Detectors: schema violation, hallucinated entity (heuristic), refusal,
  instruction neglect, tool invocation error
- Perturbations: paraphrase (rule-based), typo injection, context inflation,
  instruction injection, schema edge cases
- Harness: audit loop with latent-inconsistency measurement, coverage metric,
  leakage metric, per-step fire-rate table
- One example pipeline: structured extraction
- Stub backend for CPU dev, HuggingFace backend for real runs

## What's TODO after reading the taxonomy paper

- Reconcile `FailureMode` enum values with the paper's exact mode names.
  Drop modes you can't mechanically detect; add any you can.
- Build pipelines 2 and 3:
  - `rag.py`: retrieve -> generate -> citation-check
  - `tool_agent.py`: plan -> select_tool -> call_tool -> verify
- Decide on the model pairs you'll commit to (see `MODEL_PAIRS` in backends.py).
- SLM-based paraphraser (use a small model to paraphrase inputs -- better than
  rule-based, but slower). Optional.
- Spot-check a sample of detector firings by hand and report precision/recall
  in the writeup. This is standard in LLM-eval papers and protects you from
  the "your detectors are noisy" reviewer comment.
- Plotting / analysis script that reads the JSON dumps and produces:
  - failure-mode histogram per (pipeline, model)
  - small-vs-large difference plot (the headline figure)
  - declared-coverage vs leakage scatter

## Headline metrics for the report

For each (pipeline, model_size) combination:
- per-step failure-mode firing rate
- declared coverage: fraction of declared modes that fired at least once
- leakage rate: fraction of firings that were undeclared
- latent inconsistency rate: fraction of seeds where reruns produced
  distinct outputs at some step

The headline comparison: small vs. large failure-mode distribution difference,
per pipeline. Hypothesis: small models lean toward SCHEMA_VIOLATION,
HALLUCINATED_ENTITY, INSTRUCTION_NEGLECT; large models lean toward more
subtle modes like REASONING_DRIFT.

If the hypothesis is wrong, the negative result is still publishable as
"small models, when used in well-designed pipelines, exhibit failure modes
similar to large models -- contrary to expectation."
