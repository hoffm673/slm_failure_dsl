"""
The audit harness: run a pipeline across many perturbed inputs and aggregate
failure-mode firing rates per step and per perturbation strategy.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

from .perturbations import PERTURBATIONS, Perturbation

if TYPE_CHECKING:
    from .core import Pipeline
    from .runtime import RunTrace


# Failure modes that are detected at the harness level across reruns, not
# declared per-step. Excluded from declared_coverage and leakage_rate so those
# metrics measure what they actually mean.
HARNESS_LEVEL_MODES = {"latent_inconsistency"}


@dataclass
class AuditReport:
    pipeline_name: str
    model_name: str
    n_seeds: int
    n_perturbations_per_seed: int

    fire_counts: dict[tuple[str, str], int] = field(default_factory=dict)
    total_runs: int = 0

    declared_per_step: dict[str, set[str]] = field(default_factory=dict)
    fired_per_step: dict[str, set[str]] = field(default_factory=dict)

    inconsistency_counts: dict[tuple[str, int], int] = field(default_factory=dict)

    sample_traces: list = field(default_factory=list)

    def fire_rate(self, step: str, mode: str) -> float:
        if self.total_runs == 0:
            return 0.0
        return self.fire_counts.get((step, mode), 0) / self.total_runs

    def declared_coverage(self) -> float:
        """Fraction of declared step-level failure modes that fired at least once."""
        total = 0
        covered = 0
        for step, declared in self.declared_per_step.items():
            fired = self.fired_per_step.get(step, set()) - HARNESS_LEVEL_MODES
            total += len(declared)
            covered += len(declared & fired)
        return covered / total if total else 1.0

    def leakage_rate(self) -> float:
        """Fraction of step-level firings whose mode was not declared on that step."""
        leaked = 0
        total_fired = 0
        for step, fired in self.fired_per_step.items():
            declared = self.declared_per_step.get(step, set())
            fired_step_level = fired - HARNESS_LEVEL_MODES
            leaked += len(fired_step_level - declared)
            total_fired += len(fired_step_level)
        return leaked / total_fired if total_fired else 0.0

    def inconsistency_rate(self) -> float:
        """Fraction of (step, seed) combinations where reruns disagreed."""
        if self.n_seeds == 0:
            return 0.0
        total_combos = self.n_seeds * len(self.declared_per_step)
        return len(self.inconsistency_counts) / total_combos if total_combos else 0.0

    def to_dict(self) -> dict:
        return {
            "pipeline": self.pipeline_name,
            "model": self.model_name,
            "n_seeds": self.n_seeds,
            "n_perturbations_per_seed": self.n_perturbations_per_seed,
            "total_runs": self.total_runs,
            "fire_counts": {f"{s}|{m}": c for (s, m), c in self.fire_counts.items()},
            "declared_per_step": {k: sorted(v) for k, v in self.declared_per_step.items()},
            "fired_per_step": {k: sorted(v) for k, v in self.fired_per_step.items()},
            "declared_coverage": self.declared_coverage(),
            "leakage_rate": self.leakage_rate(),
            "inconsistency_rate": self.inconsistency_rate(),
            "sample_traces": self.sample_traces,
        }


def audit_pipeline(
    pipeline: "Pipeline",
    seed_inputs: list[dict],
    model_name: str = "unknown",
    n_perturbations_per_seed: int = 10,
    rerun_for_consistency: int = 3,
    perturbations: list[Perturbation] | None = None,
    rng_seed: int = 0,
) -> AuditReport:
    perturbations = perturbations or PERTURBATIONS
    rng = random.Random(rng_seed)

    report = AuditReport(
        pipeline_name=pipeline.name,
        model_name=model_name,
        n_seeds=len(seed_inputs),
        n_perturbations_per_seed=n_perturbations_per_seed,
    )

    for step_name, step in pipeline.steps.items():
        report.declared_per_step[step_name] = {f.value for f in step.effects.may_fail}
        report.fired_per_step[step_name] = set()

    for seed_idx, seed_input in enumerate(seed_inputs):
        reruns = []
        for _ in range(rerun_for_consistency):
            trace = pipeline.run(seed_input)
            reruns.append(trace)
            _record_trace(report, trace)
        _measure_inconsistency(report, reruns, seed_idx)

        for pert in perturbations:
            for _ in range(n_perturbations_per_seed):
                perturbed = _apply_perturbation(seed_input, pert, rng)
                trace = pipeline.run(perturbed)
                _record_trace(report, trace)

        if seed_idx < 5:
            report.sample_traces.append({
                "seed_idx": seed_idx,
                "input": seed_input,
                "trace": [_trace_step_summary(s) for s in reruns[0].steps],
            })

    return report


def _apply_perturbation(seed: dict, pert: Perturbation, rng: random.Random) -> dict:
    perturbed = dict(seed)
    for k, v in seed.items():
        if isinstance(v, str):
            perturbed[k] = pert.apply(v, rng)
            break
    return perturbed


def _record_trace(report: AuditReport, trace: "RunTrace") -> None:
    report.total_runs += 1
    for step_trace in trace.steps:
        for mode in step_trace.fired_failures:
            key = (step_trace.step_name, mode)
            report.fire_counts[key] = report.fire_counts.get(key, 0) + 1
            report.fired_per_step[step_trace.step_name].add(mode)


def _measure_inconsistency(report: AuditReport, reruns: list, seed_idx: int) -> None:
    per_step_outputs: dict[str, set] = defaultdict(set)
    for trace in reruns:
        for step_trace in trace.steps:
            try:
                rep = str(step_trace.output)
                per_step_outputs[step_trace.step_name].add(rep)
            except Exception:
                pass
    for step_name, outs in per_step_outputs.items():
        if len(outs) > 1:
            report.inconsistency_counts[(step_name, seed_idx)] = len(outs)
            report.fired_per_step[step_name].add("latent_inconsistency")
            key = (step_name, "latent_inconsistency")
            report.fire_counts[key] = report.fire_counts.get(key, 0) + 1


def _trace_step_summary(step_trace) -> dict:
    return {
        "step": step_trace.step_name,
        "fired": step_trace.fired_failures,
        "latency_ms": round(step_trace.latency_ms, 2),
        "output": _safe_repr(step_trace.output),
        "raw_text": step_trace.raw_text,
    }


def _safe_repr(x):
    try:
        s = repr(x)
        return s[:500] + "..." if len(s) > 500 else s
    except Exception:
        return "<unrepresentable>"