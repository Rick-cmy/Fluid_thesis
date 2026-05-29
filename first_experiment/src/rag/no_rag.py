from rag.base import RAGRetriever, TermHit


class NoRAG(RAGRetriever):
    """Baseline: no retrieval. Recall@k = 0 by definition."""

    def retrieve(self, source: str, draft: str, k: int = 5) -> list[TermHit]:
        return []
