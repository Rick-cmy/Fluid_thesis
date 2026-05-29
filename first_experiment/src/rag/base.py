from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class TermHit:
    sv: str
    en: str
    score: float
    source: str = "client"

    def to_context_line(self) -> str:
        return f"  • {self.sv} → {self.en}"


class RAGRetriever(ABC):

    @abstractmethod
    def retrieve(self, source: str, draft: str, k: int = 5) -> list[TermHit]:
        """Return top-k terminology hits for the given source/draft pair."""

    def format_context(self, hits: list[TermHit]) -> str:
        if not hits:
            return ""
        lines = ["Relevant terminology (Swedish → English):"]
        lines += [h.to_context_line() for h in hits]
        return "\n".join(lines)

    def recall_at_k(self, source: str, gold_sv_terms: list[str], k: int = 5) -> float:
        """
        Measures recall@k: fraction of gold_sv_terms that appear in top-k hits.
        Used as the operationalised proxy for RAG grounding strength.
        """
        if not gold_sv_terms:
            return 0.0
        hits = self.retrieve(source, "", k=k)
        retrieved_sv = {h.sv.lower() for h in hits}
        matched = sum(1 for t in gold_sv_terms if t.lower() in retrieved_sv)
        return matched / len(gold_sv_terms)
