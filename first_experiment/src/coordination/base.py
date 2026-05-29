from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DimensionResult:
    agent: str
    has_issue: bool
    severity: str      # none | minor | major | critical
    issue_span: str
    suggested_revision: str
    explanation: str
    confidence: float


@dataclass
class ReviewResult:
    """Output of a full coordination run for one segment."""
    final_recommendation: str
    dimension_results: list[DimensionResult] = field(default_factory=list)
    accepted_points: list[dict[str, Any]] = field(default_factory=list)
    rejected_points: list[dict[str, Any]] = field(default_factory=list)
    conflicts: list[dict[str, Any]] = field(default_factory=list)
    reasoning_summary: str = ""
    confidence: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0


class Coordinator(ABC):

    @abstractmethod
    def run(
        self,
        source: str,
        draft: str,
        source_lang: str,
        target_lang: str,
        domain: str,
        rag_context: str,
        model: str,
    ) -> ReviewResult:
        """Run the coordination protocol and return a ReviewResult."""
