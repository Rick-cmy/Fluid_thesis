from __future__ import annotations

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from data.loader import Term
from rag.base import RAGRetriever, TermHit

_MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"


class TermRAG(RAGRetriever):
    """
    Dense vector retrieval over the termbase union.

    Embeds all sv + en forms using a multilingual sentence-transformer,
    normalises to unit length, and indexes in a FAISS flat inner-product
    index (equivalent to cosine similarity).

    Retrieval: embed the source (sv) text, fetch n_candidates from FAISS,
    deduplicate by canonical sv form, return top-k TermHits.
    """

    def __init__(self, termbase: list[Term], model_name: str = _MODEL_NAME):
        self.termbase = termbase
        self._model = SentenceTransformer(model_name, device="cpu")
        self._index: faiss.IndexFlatIP | None = None
        self._idx_to_term: list[Term] = []
        self._build_index()

    def _build_index(self) -> None:
        texts: list[str] = []
        idx_to_term: list[Term] = []

        for term in self.termbase:
            for form in term.all_sv():
                texts.append(form)
                idx_to_term.append(term)
            for form in term.all_en():
                texts.append(form)
                idx_to_term.append(term)

        embeddings = self._model.encode(
            texts,
            batch_size=256,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        embeddings = np.array(embeddings, dtype=np.float32)

        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(embeddings)
        self._idx_to_term = idx_to_term

    def retrieve(self, source: str, draft: str, k: int = 5) -> list[TermHit]:
        query = np.array(
            self._model.encode([source], normalize_embeddings=True),
            dtype=np.float32,
        )

        # Over-retrieve to allow deduplication by canonical sv form
        n_candidates = min(k * 10, self._index.ntotal)
        scores, indices = self._index.search(query, n_candidates)

        seen_sv: set[str] = set()
        hits: list[TermHit] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            term = self._idx_to_term[idx]
            if term.sv in seen_sv:
                continue
            seen_sv.add(term.sv)
            hits.append(TermHit(sv=term.sv, en=term.en, score=float(score), source=term.source))
            if len(hits) >= k:
                break

        return hits
