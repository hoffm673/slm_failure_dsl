"""
Pipeline execution + RunTrace data structure.

The runtime walks the topo-sorted DAG, calls each step's function, and records
a structured trace. The trace is the unit of analysis for the harness: it tells
us which failure modes fired at which steps.

Step functions take a `context` dict (containing prior step outputs keyed by step
name) and return either:
  - a plain value (treated as the step's output, no failures fired)
  - a StepResult (full control over output + fired failures + metadata)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .core import StepResult

if TYPE_CHECKING:
    from .core import FailureMode, Pipeline


@dataclass
class StepTrace:
    step_name: str
    output: Any
    fired_failures: list[str] = field(default_factory=list)  # FailureMode.value strings
    latency_ms: float = 0.0
    raw_text: str | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class RunTrace:
    pipeline_name: str
    initial_context: dict
    steps: list[StepTrace] = field(default_factory=list)
    total_latency_ms: float = 0.0

    def all_fired(self) -> set[str]:
        out: set[str] = set()
        for s in self.steps:
            out |= set(s.fired_failures)
        return out

    def fired_at(self, step_name: str) -> set[str]:
        for s in self.steps:
            if s.step_name == step_name:
                return set(s.fired_failures)
        return set()


def execute(pipeline: "Pipeline", initial_context: dict) -> RunTrace:
    context: dict = dict(initial_context)
    trace = RunTrace(pipeline_name=pipeline.name, initial_context=dict(initial_context))

    t0 = time.perf_counter()
    for name in pipeline._order:
        step = pipeline.steps[name]
        s0 = time.perf_counter()
        raw_result = step.fn(context)
        elapsed = (time.perf_counter() - s0) * 1000.0

        if isinstance(raw_result, StepResult):
            sr = raw_result
            sr.latency_ms = elapsed
        else:
            sr = StepResult(output=raw_result, latency_ms=elapsed)

        context[name] = sr.output
        trace.steps.append(StepTrace(
            step_name=name,
            output=sr.output,
            fired_failures=[f.value for f in sr.fired_failures],
            latency_ms=sr.latency_ms,
            raw_text=sr.raw_text,
            metadata=sr.metadata,
        ))

    trace.total_latency_ms = (time.perf_counter() - t0) * 1000.0
    return trace
