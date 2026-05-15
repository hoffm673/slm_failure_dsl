"""
Real experiment runner.

For each (pipeline, model) pair, run the audit and dump the report as JSON.
Plot scripts (e.g. notebook) read the JSON.

Run with:
    python -m experiments.run_experiment --model_id Qwen/Qwen2.5-0.5B-Instruct
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from slm_failure_dsl import TransformersModel, StubModel, audit_pipeline


def get_pipeline(name: str, model):
    if name == "structured_extraction":
        from examples.structured_extraction import build_pipeline, SEED_INPUTS
        return build_pipeline(model), SEED_INPUTS
    raise ValueError(f"unknown pipeline: {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", default="structured_extraction")
    ap.add_argument("--model_id", required=True,
                    help="HuggingFace model id, or 'stub' for the deterministic stub")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n_perturbations", type=int, default=10)
    ap.add_argument("--n_reruns", type=int, default=3)
    ap.add_argument("--out_dir", default="results")
    args = ap.parse_args()

    if args.model_id == "stub":
        model = StubModel(name="stub")
    else:
        model = TransformersModel(args.model_id, device=args.device)

    pipeline, seeds = get_pipeline(args.pipeline, model)
    pipeline.compile()

    report = audit_pipeline(
        pipeline,
        seed_inputs=seeds,
        model_name=model.name,
        n_perturbations_per_seed=args.n_perturbations,
        rerun_for_consistency=args.n_reruns,
    )

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    safe = args.model_id.replace("/", "__")
    out = Path(args.out_dir) / f"{args.pipeline}__{safe}.json"
    with out.open("w") as f:
        json.dump(report.to_dict(), f, indent=2)

    print(f"wrote {out}")
    print(f"declared coverage: {report.declared_coverage():.2%}")
    print(f"leakage rate     : {report.leakage_rate():.2%}")


if __name__ == "__main__":
    main()
