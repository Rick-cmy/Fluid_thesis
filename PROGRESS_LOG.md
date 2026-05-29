# Thesis Progress Log — Mingyang Chen

**Project:** Retrieval or Coordination? An Error-Type Analysis of When RAG Terminology Grounding Can Reduce Multi-Agent Coordination in Financial Translation QA (Swedish–English REIT Case Study)

**Supervisors:** Prof. Jörg Conradt (KTH), Prof. Arvind Kumar (examiner), Gabriella Rudström (Fluid Translation)

---

## 2026-05-29 — End of May update

### What was built this month

The full controlled experiment framework (`first_experiment/`) is now functional end-to-end:

| Component | File | Status |
|-----------|------|--------|
| Data loader | `src/data/loader.py` | ✅ Done |
| TM-match splitter (80/20 dev/test) | `src/data/splitter.py` | ✅ Done |
| Error injector (6 types) | `src/data/injector.py` | ✅ Done |
| No-RAG retriever (baseline) | `src/rag/no_rag.py` | ✅ Done |
| SingleAgent coordinator | `src/coordination/single_agent.py` | ✅ Done |
| Pipeline coordinator (7× parallel) | `src/coordination/pipeline.py` | ✅ Done |
| Debate coordinator (CHORUS, 8× parallel) | `src/coordination/debate.py` | ✅ Done |
| Metrics (P/R/F1 per error type) | `src/eval/metrics.py` | ✅ Done |
| Cost tracker (tokens + latency) | `src/eval/cost.py` | ✅ Done |
| LLM client (Ollama + external API) | `src/llm.py` | ✅ Done |
| Baseline runner | `scripts/run_baseline.py` | ✅ Done |
| Full grid runner | `scripts/run_grid.py` | ✅ Done |
| Smoke test (all 3 coordinators) | `scripts/smoke_test_parallel.py` | ✅ Done |
| VectorRAG (sentence-transformers + FAISS) | `src/rag/vector_rag.py` | 🔜 Next |
| HybridRAG (BM25 + dense) | `src/rag/hybrid_rag.py` | 🔜 Pending |
| xCOMET span scorer | `src/eval/xcomet.py` | 🔜 Pending |

### Baseline result (C1: no_rag × single_agent)

- Model: qwen3:8b (Ollama, CPU)
- n = 5 segments per error type (25 total, smoke-test scale)

```
Macro F1:      0.762
Mean latency:  49 s/segment (CPU — GPU/API expected ~1–3 s)
Tokens:        ~782/segment

  terminology    P=1.00  R=0.80  F1=0.889
  numeracy       P=1.00  R=0.80  F1=0.889
  fluency        P=1.00  R=0.40  F1=0.571  ← coordination-addressable
  style_guide    P=1.00  R=0.40  F1=0.571  ← coordination-addressable
  consistency    P=1.00  R=0.80  F1=0.889
```

**Key observation:** Precision = 1.0 across all types — zero false alarms. Recall is the discriminating dimension. Fluency and style_guide show substantially lower recall (0.40 vs. 0.80), which is exactly the direction the hypothesis predicts: a single agent without cross-review or style-guide context struggles with coordination-addressable error types.

> **Note — `named_entity` now implemented (2026-05-29):** Added `inject_named_entity` and `_NE_SWAPS` to `injector.py`, and wired it into `build_eval_set`. Swap pairs cover company names (Catena↔WDP), Swedish cities (Stockholm↔Gothenburg, Norrköping↔Jönköping), market segment (Large Cap↔Mid Cap), stock exchange, and certification standards. Will appear in results from next baseline run.

### Coordinator smoke test (3 segments, qwen3:8b, CPU)

```
                   Latency (avg)   Tokens/seg   Fluency error detected?
SingleAgent        52 s            ~809         ✗ missed
Pipeline (7×par)   89 s            ~3,225       ✓ caught
Debate   (8×par)   151 s           ~4,912       ✓ caught
```

**Key observation:** The single agent completely missed the injected fluency error; both multi-agent coordinators caught it. This is the first concrete evidence supporting the core hypothesis — fluency errors require coordination, not just retrieval.

> **Hardware note:** Pipeline/Debate parallelism has no speedup on CPU (Ollama processes one request at a time). True speedup requires GPU or a cloud API. All latency comparisons in the final results will use Other LLM API (~1–3 s/segment).

### Literature

76 of 80 target papers downloaded to `papers/`. Full architecture design documented in `agents_architecture.md`.

---

## Next steps (June)

1. Implement `VectorRAG` — sentence-transformers embeddings over the 1,938-term Catena termbase, FAISS index, top-5 retrieval
2. Run full baseline at scale: `--n-per-type 200` via Other LLM API
3. Run 9-cell grid (3 RAG × 3 coordination) for the no_rag and vector_rag rows (6 of 9 cells)
4. Begin error-type breakdown analysis comparing C1 vs. C2–C4

---
