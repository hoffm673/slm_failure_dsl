"""
Compile-time static analysis over the pipeline DAG.

This is the "compiler" part of the DSL. It runs before execution and produces
a CompileReport with warnings about:
  - undeclared dependencies (step references a name that doesn't exist)
  - cycles
  - unhandled effects (failures declared upstream but never recovered)
  - dead recoveries (a step declares it recovers X but no upstream step may emit X)

The undeclared-effect leakage metric in the eval comes from comparing the static
analysis's predicted unhandled set against what the harness actually observes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .core import FailureMode, Pipeline


@dataclass
class CompileReport:
    topo_order: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    # for each step, the set of failure modes that could reach it unhandled
    propagated_failures: dict[str, set] = field(default_factory=dict)
    # failure modes that escape the pipeline (reach a leaf without being recovered)
    unhandled_at_boundary: set = field(default_factory=set)

    def summary(self) -> str:
        lines = [f"Compile report: {len(self.topo_order)} steps"]
        if self.warnings:
            lines.append("Warnings:")
            lines.extend(f"  - {w}" for w in self.warnings)
        if self.unhandled_at_boundary:
            modes = sorted(m.value for m in self.unhandled_at_boundary)
            lines.append(f"Unhandled at boundary: {modes}")
        return "\n".join(lines)


def _topo_sort(pipeline: "Pipeline") -> tuple[list[str], list[str]]:
    """Kahn's algorithm. Returns (order, warnings)."""
    warnings: list[str] = []
    in_degree: dict[str, int] = {n: 0 for n in pipeline.steps}
    adj: dict[str, list[str]] = {n: [] for n in pipeline.steps}

    for name, step in pipeline.steps.items():
        for dep in step.depends_on:
            if dep not in pipeline.steps:
                warnings.append(f"step {name!r} depends on unknown step {dep!r}")
                continue
            adj[dep].append(name)
            in_degree[name] += 1

    queue = [n for n, d in in_degree.items() if d == 0]
    order: list[str] = []
    while queue:
        n = queue.pop(0)
        order.append(n)
        for m in adj[n]:
            in_degree[m] -= 1
            if in_degree[m] == 0:
                queue.append(m)

    if len(order) != len(pipeline.steps):
        remaining = [n for n in pipeline.steps if n not in order]
        warnings.append(f"cycle detected involving: {remaining}")
        order.extend(remaining)  # so downstream code doesn't crash

    return order, warnings


def compile_pipeline(pipeline: "Pipeline") -> CompileReport:
    order, warnings = _topo_sort(pipeline)
    propagated: dict[str, set] = {n: set() for n in pipeline.steps}

    # forward dataflow: propagate may_fail downstream, subtracting recoveries
    for name in order:
        step = pipeline.steps[name]
        # collect failures arriving from upstream
        incoming: set = set()
        for dep in step.depends_on:
            if dep in propagated:
                incoming |= propagated[dep]
        # this step may itself emit its declared may_fail set
        incoming |= set(step.effects.may_fail)
        # this step recovers some -- subtract them
        recovered = set(step.recovers)
        dead_recoveries = recovered - incoming
        if dead_recoveries:
            modes = sorted(m.value for m in dead_recoveries)
            warnings.append(
                f"step {name!r} declares recovery for {modes} but no upstream "
                f"step can emit those"
            )
        propagated[name] = incoming - recovered

    # leaves = nodes with no outgoing edges
    has_outgoing: set[str] = set()
    for s in pipeline.steps.values():
        for dep in s.depends_on:
            has_outgoing.add(dep)
    leaves = [n for n in pipeline.steps if n not in has_outgoing]

    unhandled: set = set()
    for leaf in leaves:
        unhandled |= propagated[leaf]

    return CompileReport(
        topo_order=order,
        warnings=warnings,
        propagated_failures=propagated,
        unhandled_at_boundary=unhandled,
    )
