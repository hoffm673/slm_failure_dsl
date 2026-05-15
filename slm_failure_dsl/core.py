"""
Core DSL primitives: Pipeline, Step, FailureMode, EffectSignature.

A Step is a unit of LLM (or pure) computation with a declared effect signature.
A Pipeline is a DAG of Steps. The compiler walks the DAG to verify that every
declared failure mode is either handled by a `recover` clause downstream, or
propagates to the pipeline boundary.

Design notes:
  - Effect names follow the Microsoft (2025) system-level failure-mode taxonomy.
    Pick the subset you can actually detect after reading the paper.
  - We use decorators + a Pipeline builder rather than a parsed external syntax.
    This is an embedded DSL -- the host language is Python.
  - Steps are pure-ish: they take a dict (context) and return a dict (updates).
    This makes DAG analysis trivial and avoids global state.
"""

from __future__ import annotations

import enum
import functools
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


class FailureMode(enum.Enum):
    """
    Failure modes drawn from the Microsoft taxonomy (Vinay 2025, arXiv:2511.19933).
    The taxonomy lists 15; start with the subset that is mechanically detectable
    on small models without human judgment.

    TODO after reading paper: prune / rename to match exact taxonomy language.
    """
    SCHEMA_VIOLATION = "schema_violation"               # output doesn't match declared schema
    REASONING_DRIFT = "reasoning_drift"                 # multi-step answer diverges from prior steps
    LATENT_INCONSISTENCY = "latent_inconsistency"       # same input -> different outputs across runs
    CONTEXT_BOUNDARY_DEGRADATION = "context_boundary"   # quality collapses near context-length limit
    TOOL_INVOCATION_ERROR = "tool_invocation_error"     # bad tool name / bad args
    HALLUCINATED_ENTITY = "hallucinated_entity"         # output references entity not in input/context
    INSTRUCTION_NEGLECT = "instruction_neglect"         # explicit instruction in prompt ignored
    REFUSAL = "refusal"                                 # model declines benign request
    ENTITY_OMISSION       = "entity_omission"
    COMMON_NOUN_AS_ENTITY = "common_noun_as_entity"


@dataclass
class EffectSignature:
    """What a step is allowed to fail with, and what it must produce on success."""
    may_fail: list[FailureMode] = field(default_factory=list)
    output_schema: Optional[dict] = None  # JSON-schema-style dict, optional
    description: str = ""


@dataclass
class StepResult:
    """Outcome of running a single step."""
    output: Any
    fired_failures: list[FailureMode] = field(default_factory=list)
    raw_text: Optional[str] = None        # for LLM steps, the unparsed model output
    latency_ms: float = 0.0
    metadata: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return len(self.fired_failures) == 0


class Step:
    """A node in the pipeline DAG."""

    def __init__(
        self,
        name: str,
        fn: Callable,
        effects: EffectSignature,
        depends_on: Optional[list[str]] = None,
        recovers: Optional[list[FailureMode]] = None,
    ):
        self.name = name
        self.fn = fn
        self.effects = effects
        self.depends_on = depends_on or []
        self.recovers = recovers or []

    def __repr__(self) -> str:
        return f"Step({self.name}, may_fail={[f.value for f in self.effects.may_fail]})"


def step(
    name: Optional[str] = None,
    may_fail: Optional[list[FailureMode]] = None,
    output_schema: Optional[dict] = None,
    depends_on: Optional[list[str]] = None,
    recovers: Optional[list[FailureMode]] = None,
):
    """Decorator to mark a function as a pipeline step with an effect signature."""

    def decorator(fn: Callable) -> Step:
        step_name = name or fn.__name__
        sig = EffectSignature(
            may_fail=may_fail or [],
            output_schema=output_schema,
            description=inspect.getdoc(fn) or "",
        )
        return Step(
            name=step_name,
            fn=fn,
            effects=sig,
            depends_on=depends_on,
            recovers=recovers,
        )

    return decorator


class Pipeline:
    """
    A DAG of Steps. Built imperatively, then `.compile()` to run static checks,
    then `.run(input)` to execute, or `.audit(generator)` to run the perturbation harness.
    """

    def __init__(self, name: str):
        self.name = name
        self.steps: dict[str, Step] = {}
        self._order: list[str] = []  # topological order, set by compile()
        self._compiled = False

    def add(self, *steps: Step) -> "Pipeline":
        for s in steps:
            if s.name in self.steps:
                raise ValueError(f"duplicate step name: {s.name}")
            self.steps[s.name] = s
        return self

    def compile(self) -> "CompileReport":
        """
        Static analysis pass:
          1. topological sort, detect cycles
          2. resolve depends_on references
          3. compute effect-leakage: which declared failures are NOT recovered downstream
        Returns a CompileReport. Does not raise on warnings.
        """
        from .compiler import compile_pipeline
        report = compile_pipeline(self)
        self._order = report.topo_order
        self._compiled = True
        return report

    def run(self, initial_context: dict) -> "RunTrace":
        """Execute the pipeline once on a single input."""
        if not self._compiled:
            self.compile()
        from .runtime import execute
        return execute(self, initial_context)

    def audit(self, *args, **kwargs):
        """Run the perturbation harness. See harness.audit_pipeline for args."""
        if not self._compiled:
            self.compile()
        from .harness import audit_pipeline
        return audit_pipeline(self, *args, **kwargs)
