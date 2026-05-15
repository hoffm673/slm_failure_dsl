"""slm_failure_dsl: an effect-typed DSL + audit harness for small-LLM pipelines."""

from .core import (
    FailureMode,
    EffectSignature,
    StepResult,
    Step,
    step,
    Pipeline,
)
from .compiler import CompileReport
from .runtime import RunTrace, StepTrace
from .harness import AuditReport, audit_pipeline
from .backends import ModelBackend, StubModel, TransformersModel, MODEL_PAIRS
from . import detectors
from . import perturbations

__all__ = [
    "FailureMode",
    "EffectSignature",
    "StepResult",
    "Step",
    "step",
    "Pipeline",
    "CompileReport",
    "RunTrace",
    "StepTrace",
    "AuditReport",
    "audit_pipeline",
    "ModelBackend",
    "StubModel",
    "TransformersModel",
    "MODEL_PAIRS",
    "detectors",
    "perturbations",
]
