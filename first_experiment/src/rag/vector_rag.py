from __future__ import annotations

from data.loader import Term
from rag.base import RAGRetriever, TermHit

# July implementation: sentence-transformers + FAISS index over termbase union
# Scaffold only — raises NotImplementedError until implemented.


class VectorRAG(RAGRetriever):
    """
    Dense vector retrieval over the termbase union
    (1,938 Catena terms + 1,328 IFRS/IAS terms).

    Uses sentence-transformers/paraphrase-multilingual-mpnet-base-v2
    for bilingual sv+en embeddings, indexed in FAISS.

    July implementation task:
        pip install sentence-transformers faiss-cpu
        Build index: embed all (sv + en) forms, store in FAISS flat L2 index.
        Retrieve: embed query (source sv text), return top-k Term hits.
    """

    def __init__(self, termbase: list[Term]):
        self.termbase = termbase
        self._index = None
        self._build_index()

    def _build_index(self) -> None:
        raise NotImplementedError(
            "VectorRAG index not yet built. "
            "Install sentence-transformers + faiss-cpu and implement July task."
        )

    def retrieve(self, source: str, draft: str, k: int = 5) -> list[TermHit]:
        raise NotImplementedError("VectorRAG not implemented — July task.")
