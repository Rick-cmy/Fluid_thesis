from __future__ import annotations

from data.loader import Term
from rag.base import RAGRetriever, TermHit

# August implementation: BM25 (rank_bm25) + dense re-rank
# Scaffold only — raises NotImplementedError until implemented.


class HybridRAG(RAGRetriever):
    """
    Hybrid retrieval: BM25 candidate generation + dense re-ranking.

    Step 1 (BM25): tokenise source, retrieve top-50 candidates from termbase.
    Step 2 (dense re-rank): embed candidates + query, take top-k by cosine sim.

    This achieves higher recall@k than vector-only for exact Swedish term matches
    (BM25 handles OOV / morphological variants well).

    August implementation task:
        pip install rank_bm25 sentence-transformers
        Build BM25 corpus over sv+sv_variants of all terms.
        Re-rank with the same multilingual encoder as VectorRAG.
    """

    def __init__(self, termbase: list[Term]):
        self.termbase = termbase
        self._bm25 = None
        self._build_index()

    def _build_index(self) -> None:
        raise NotImplementedError(
            "HybridRAG index not yet built. "
            "Install rank_bm25 + sentence-transformers and implement August task."
        )

    def retrieve(self, source: str, draft: str, k: int = 5) -> list[TermHit]:
        raise NotImplementedError("HybridRAG not implemented — August task.")
