from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rag.base import TermHit
from rag.term_rag import TermRAG

# Sentinel value used in TermHit.sv to distinguish style-rule hits from term hits
_STYLE_SOURCE = "style_guide"


class RichRAG(TermRAG):
    """
    Rich retrieval: termbase (sv→en) + Client style-guide rules.

    Extends TermRAG identically for terminology retrieval (queries the Swedish
    source text). Adds a second FAISS index over style-guide rules (queries the
    English draft text), so that rules are surfaced when the draft may contain
    wrong forms.

    retrieve() returns up to k term hits + up to k_rules style hits.
    format_context() produces two clearly labelled sections.

    This tests the thesis claim that mechanical style-guide violations become
    RAG-addressable when the style rules are in the knowledge base.
    """

    def __init__(
        self,
        termbase,
        style_rules_path: str | Path,
        model_name: str = "paraphrase-multilingual-mpnet-base-v2",
        k_rules: int = 3,
    ):
        super().__init__(termbase, model_name=model_name)
        self.k_rules = k_rules
        self._style_rules: list[dict] = []
        self._rule_index = None
        self._load_style_rules(style_rules_path)

    def _load_style_rules(self, path: str | Path) -> None:
        import faiss

        rules = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rules.append(json.loads(line))

        if not rules:
            return

        texts = [r["embed_text"] for r in rules]
        embeddings = self._model.encode(
            texts,
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        embeddings = np.array(embeddings, dtype=np.float32)

        dim = embeddings.shape[1]
        self._rule_index = faiss.IndexFlatIP(dim)
        self._rule_index.add(embeddings)
        self._style_rules = rules

    def _retrieve_rules(self, draft: str, k: int) -> list[TermHit]:
        if self._rule_index is None or not draft.strip():
            return []

        n = min(k, self._rule_index.ntotal)
        query = np.array(
            self._model.encode([draft], normalize_embeddings=True),
            dtype=np.float32,
        )
        scores, indices = self._rule_index.search(query, n)

        hits: list[TermHit] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            rule = self._style_rules[idx]
            hits.append(
                TermHit(
                    sv=rule["wrong"],
                    en=rule["correct"],
                    score=float(score),
                    source=_STYLE_SOURCE,
                )
            )
        return hits

    def retrieve(self, source: str, draft: str, k: int = 5) -> list[TermHit]:
        term_hits = super().retrieve(source, draft, k=k)
        rule_hits = self._retrieve_rules(draft, k=self.k_rules)
        return term_hits + rule_hits

    def format_context(self, hits: list[TermHit]) -> str:
        term_hits = [h for h in hits if h.source != _STYLE_SOURCE]
        rule_hits = [h for h in hits if h.source == _STYLE_SOURCE]

        sections: list[str] = []

        if term_hits:
            lines = ["Relevant terminology (Swedish → English):"]
            lines += [h.to_context_line() for h in term_hits]
            sections.append("\n".join(lines))

        if rule_hits:
            lines = ["Client style-guide rules (apply to English draft):"]
            for h in rule_hits:
                lines.append(f"  • Use '{h.en}' not '{h.sv}'")
            sections.append("\n".join(lines))

        return "\n\n".join(sections)
