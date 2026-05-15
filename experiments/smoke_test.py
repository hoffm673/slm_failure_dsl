"""
Smoke test: build the structured-extraction pipeline with the stub model,
compile, run, and audit. Should run in < 1 second on CPU with no dependencies
beyond stdlib.

Use this to verify the DSL/harness wiring before plugging in a real model.

Run with:
    cd /home/claude/slm_failure_dsl
    python -m experiments.smoke_test
"""

from __future__ import annotations

import json

from slm_failure_dsl import StubModel, audit_pipeline
from examples.structured_extraction import SEED_INPUTS, build_pipeline


def main():
    print("=== building pipeline with StubModel ===")
    model = StubModel(name="stub", seed=42, failure_rate=0.2)
    pipeline = build_pipeline(model)

    print("\n=== compile pass ===")
    report = pipeline.compile()
    print(report.summary())

    print("\n=== single run ===")
    trace = pipeline.run(SEED_INPUTS[0])
    for st in trace.steps:
        print(f"  {st.step_name:12s} fired={st.fired_failures} latency_ms={st.latency_ms:.1f}")
        print(f"               output={st.output}")

    print("\n=== audit (small sample) ===")
    audit = audit_pipeline(
        pipeline,
        seed_inputs=SEED_INPUTS,
        model_name=model.name,
        n_perturbations_per_seed=5,
        rerun_for_consistency=3,
    )
    print(f"  total runs       : {audit.total_runs}")
    print(f"  declared coverage: {audit.declared_coverage():.2%}")
    print(f"  leakage rate     : {audit.leakage_rate():.2%}")
    print("  per-step fire rates:")
    for (step_name, mode), count in sorted(audit.fire_counts.items()):
        rate = count / audit.total_runs
        declared = mode in audit.declared_per_step.get(step_name, set())
        flag = " " if declared else "*"
        print(f"    {flag} {step_name:12s} {mode:30s} {rate:.2%} ({count} firings)")
    print("  (* = undeclared / leaked)")


if __name__ == "__main__":
    main()
