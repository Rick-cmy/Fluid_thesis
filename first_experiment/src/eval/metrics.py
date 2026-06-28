from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from data.injector import ErrorType
from coordination.base import ReviewResult


@dataclass
class ErrorTypeMetrics:
    error_type: str
    tp: int = 0   # correctly flagged has_issue=True
    fp: int = 0   # flagged but no real issue (false alarm on clean segments)
    fn: int = 0   # missed: real issue not flagged
    tn: int = 0   # correctly quiet on clean segments
    critical_tp: int = 0  # critical-severity issue correctly caught
    critical_total: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def critical_catch_rate(self) -> float:
        return self.critical_tp / self.critical_total if self.critical_total > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "error_type": self.error_type,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "critical_catch_rate": round(self.critical_catch_rate, 4),
            "tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn,
        }


@dataclass
class CellMetrics:
    rag_level: str
    coord_level: str
    trial: int
    per_type: dict[str, ErrorTypeMetrics] = field(default_factory=dict)

    @property
    def macro_recall(self) -> float:
        types = [m for m in self.per_type.values() if (m.tp + m.fn) > 0]
        return sum(m.recall for m in types) / len(types) if types else 0.0

    @property
    def macro_f1(self) -> float:
        """Primary metric. Eval set contains both injected (positive) and clean (negative) segments,
        so FP is measurable and F1 is meaningful."""
        types = [m for m in self.per_type.values() if (m.tp + m.fn) > 0]
        return sum(m.f1 for m in types) / len(types) if types else 0.0

    def to_dict(self) -> dict:
        return {
            "rag_level": self.rag_level,
            "coord_level": self.coord_level,
            "trial": self.trial,
            "macro_f1": round(self.macro_f1, 4),      # primary metric
            "macro_recall": round(self.macro_recall, 4),
            "per_type": {k: v.to_dict() for k, v in self.per_type.items()},
        }


# Error type → dimension name mapping (1:1, unified across all coordinators).
# All three coordinators (single_agent, pipeline, debate) use the same 6 names.
_ERROR_TYPE_TO_DIMENSION: dict[str, list[str]] = {
    ErrorType.TERMINOLOGY:  ["terminology"],
    ErrorType.NUMERACY:     ["numeracy"],
    ErrorType.NAMED_ENTITY: ["named_entity"],
    ErrorType.FLUENCY:      ["fluency"],
    ErrorType.STYLE_GUIDE:  ["style_guide"],
    ErrorType.CONSISTENCY:  ["consistency"],
}


def score_result(
    result: ReviewResult,
    error_type: ErrorType,
    has_real_issue: bool = True,
) -> tuple[bool, bool]:
    """
    Determines whether the coordinator correctly detected the injected error.

    Returns (detected: bool, any_critical_flagged: bool).

    Detection criterion: at least one dimension result relevant to this error type
    has has_issue=True with severity in {minor, major, critical}.
    """
    relevant_dims = _ERROR_TYPE_TO_DIMENSION.get(error_type, [])
    detected = any(
        d.has_issue and d.severity != "none"
        for d in result.dimension_results
        if d.agent in relevant_dims
    )
    critical_flagged = any(
        d.has_issue and d.severity == "critical"
        for d in result.dimension_results
        if d.agent in relevant_dims
    )
    return detected, critical_flagged


def accumulate(
    cell_metrics: CellMetrics,
    error_type: ErrorType,
    result: ReviewResult,
    has_real_issue: bool = True,
) -> None:
    """Updates CellMetrics in-place for one evaluated segment."""
    if error_type.value not in cell_metrics.per_type:
        cell_metrics.per_type[error_type.value] = ErrorTypeMetrics(error_type=error_type.value)

    m = cell_metrics.per_type[error_type.value]
    detected, critical_flagged = score_result(result, error_type, has_real_issue)

    if has_real_issue:
        m.critical_total += 1
        if detected:
            m.tp += 1
            if critical_flagged:
                m.critical_tp += 1
        else:
            m.fn += 1
    else:
        if detected:
            m.fp += 1
        else:
            m.tn += 1
