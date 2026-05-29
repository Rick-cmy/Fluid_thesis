# Agent Architecture: RAG × Coordination Grid for Financial Translation QA

**Thesis:** Retrieval or Coordination? An Error-Type Analysis of When RAG Terminology Grounding Can Reduce Multi-Agent Coordination in Financial Translation QA (Swedish–English REIT Case Study)  
**Author:** Mingyang Chen  
**Date:** May 2026  
**Designed as:** Anthropic AI Researcher perspective — emphasis on clean experimental control, reproducibility, and falsifiable hypothesis structure.

---

## 1. Research Question Mapping

| RQ | What it requires architecturally |
|----|----------------------------------|
| RQ1 — Error distribution across types | Evaluation harness that scores per error-type; controlled injection set |
| RQ2 — Interaction: RAG strength × coordination marginal gain | Full 3×3 grid with per-error-type F1 per cell; marginal-gain curves |
| RQ3 — Accuracy–latency–cost frontier | Token counter + wall-clock timer per cell; Pareto-frontier plot |

**Core hypothesis (falsifiable):** For retrieval-addressable error types (terminology, named-entity, numeracy), F1 gain from adding coordination decreases as RAG grounding strength increases. For coordination-addressable types (fluency, consistency, style-guide), coordination gain is RAG-independent.

---

## 2. Two Control Axes

```
RAG Grounding Strength
        │
   hybrid ──────────────────── C3  C6  C9
        │                      │   │   │
  vector ──────────────────── C2  C5  C8
        │                      │   │   │
  no_rag ──────────────────── C1  C4  C7
        └──────────────────────────────────►
                       single  pipeline  debate
                         Coordination Intensity
```

**9-cell grid.** Each cell (Cx) is one complete experimental configuration.

### RAG levels (operationalised by recall@k proxy)

| Level | Description | Recall@k target |
|-------|-------------|-----------------|
| `no_rag` | No retrieval — agent sees source+draft only | ~0% |
| `vector` | Dense retrieval over termbase union (1,938 Client + ~1,328 IFRS) | target ≥ 60% |
| `hybrid` | Dense + BM25 re-rank over termbase union | target ≥ 80% |

RAG grounding strength is **not** treated as "weak/strong" informally — it is the measurable recall@k over the termbase union at retrieval time.

### Coordination levels

| Level | Description | Agent calls per segment |
|-------|-------------|------------------------|
| `single_agent` | One LLM call with a full-review prompt across all MQM dimensions | 1 |
| `pipeline` | 7 specialist agents run independently, results aggregated by a rule-based merger (no cross-review) | 7 |
| `debate` | CHORUS: Round 1 (7 parallel specialists) → Round 2 (coordinator synthesises, resolves conflicts) | 7 + 1 = 8 |

The `pipeline` level isolates the effect of specialisation without communication — this is the key middle point between single-agent and debate.

---

## 3. Error-Type Taxonomy

Errors are pre-classified at injection time. This makes partial-substitution directly testable.

| Error Type | Category | Served by RAG? | Example |
|------------|----------|---------------|---------|
| `terminology` | retrieval-addressable | Yes | "depreciation" instead of "impairment" |
| `numeracy` | retrieval-addressable | Partially | "SEK 5.2 thousand" instead of "SEK 5.2 million" |
| `named_entity` | retrieval-addressable | Yes | wrong company name, ISIN code |
| `fluency` | coordination-addressable | No | grammatical errors, awkward phrasing |
| `style_guide` | coordination-addressable | No | "percent" vs "per cent", sentence-case headers |
| `consistency` | coordination-addressable | No | cross-segment meaning inconsistency |

---

## 4. System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         EXPERIMENT RUNNER                           │
│  grid_runner.py — iterates 9 cells × N segments × K trials         │
│  logs: per-cell per-error-type F1, token count, wall-clock latency  │
└──────┬────────────────────────────────────────┬───────────────────┘
       │                                        │
┌──────▼──────────────┐              ┌──────────▼──────────────────┐
│      RAG LAYER      │              │     COORDINATION LAYER      │
│                     │              │                             │
│  RAGRetriever(base) │              │  Coordinator(base)          │
│    ├── NoRAG        │              │    ├── SingleAgent          │
│    ├── VectorRAG    │              │    ├── Pipeline             │
│    └── HybridRAG   │              │    └── Debate (CHORUS)      │
│                     │              │                             │
│  retrieves:         │              │  SingleAgent:               │
│   terminology ctx   │              │   one prompt, all dims      │
│   from termbase     │◄─ context ──►│                             │
│   union (3,266 terms│              │  Pipeline:                  │
│   + annual reports) │              │   7 agents parallel,        │
│                     │              │   rule-based merge          │
│  measures:          │              │                             │
│   recall@k          │              │  Debate (CHORUS):           │
└─────────────────────┘              │   Round1: 7 parallel        │
                                     │   Round2: coordinator       │
                                     └─────────────────────────────┘
                                                   │
                                    ┌──────────────▼──────────────┐
                                    │     EVALUATION HARNESS      │
                                    │                             │
                                    │  xcomet.py — span scorer    │
                                    │  metrics.py — P/R/F1        │
                                    │    per error type           │
                                    │  cost.py — tokens, latency  │
                                    └─────────────────────────────┘
                                                   │
                              ┌────────────────────▼────────────────┐
                              │            DATA LAYER               │
                              │                                     │
                              │  sentence_pairs.jsonl (17,103 pairs)│
                              │  termbase.jsonl (1,938 terms)       │
                              │  benchmark/financial_qa_v0 (20 cases│
                              │  injector.py — controlled errors    │
                              │  splitter.py — TM-match vs novel    │
                              └─────────────────────────────────────┘
```

---

## 5. Component Specifications

### 5.1 Data Layer

**`loader.py`**
- Loads `sentence_pairs.jsonl` (17,103 sv→en-GB pairs)
- Loads `termbase.jsonl` (1,938 Client terms)
- Loads `benchmark/financial_qa_benchmark_v0.jsonl` (20 annotated cases)
- Outputs typed `Segment` and `Term` dataclasses

**`splitter.py`** — controls the repetition confounder
- Identifies TM exact-match segments (source appears in Q3/Q1 quarterly reports cross-quarter)
- Separates into `novel_segments` and `tm_matched_segments`
- Experiments run on `novel_segments` to avoid inflated accuracy from TM recall

**`injector.py`** — builds the per-error-type evaluation set
- Takes clean segments; injects one controlled error per segment per error type
- `inject_terminology()`: swap an IFRS term using termbase near-misses
- `inject_numeracy()`: corrupt a number/currency/date
- `inject_named_entity()`: swap a known entity
- `inject_fluency()`: introduce a grammatical error
- `inject_style_guide()`: violate Client style rules (percent/per cent, sentence case, SEK formatting)
- `inject_consistency()`: introduce a cross-segment meaning flip
- Outputs `EvalSegment(source, injected_draft, gold_translation, error_type, injected_span)`

### 5.2 RAG Layer

**`base.py`** — `RAGRetriever` abstract base
```python
class RAGRetriever(ABC):
    def retrieve(self, source: str, draft: str, k: int = 5) -> list[TermHit]:
        ...
    def recall_at_k(self, source: str, gold_terms: list[str], k: int) -> float:
        ...
```

**`no_rag.py`** — returns empty list; recall@k = 0 by definition  
**`vector_rag.py`** — embeds query against termbase using `sentence-transformers`; returns top-k  
**`hybrid_rag.py`** — BM25 (rank_bm25) + dense re-rank; returns merged top-k

The retrieved `TermHit` list is formatted as a context block injected into the agent system prompt.

### 5.3 Coordination Layer

**`single_agent.py`**  
One LLM call with all 6 error dimensions in a single prompt. Returns `{error_type: {has_issue, severity, suggested_fix}}`. Token cost: 1 call.

**`pipeline.py`**  
7 specialist agents (accuracy, terminology, fluency, style, locale_convention, audience_appropriateness, design_markup) run in parallel. Results merged: highest-severity suggestion per dimension wins. No cross-review. Token cost: 7 calls.

**`debate.py`** (adapted from `chorus_chat/src/chorus_mvp/`)  
Round 1: 7 parallel specialist agents (same as pipeline).  
Round 2: coordinator synthesises, filters invalid suggestions, resolves conflicts.  
Token cost: 8 calls. This is the full CHORUS protocol.

### 5.4 Evaluation Layer

**`xcomet.py`**  
Wraps `unbabel-comet` library. Uses `Unbabel/xCOMET-XL` (or `xCOMET-XXL` on KTH GPU).  
Inputs: `(source, hypothesis, reference)` triplets.  
Outputs: segment-level score + span-level error severity annotations.  
Supports batch inference for cost efficiency.

**`metrics.py`**  
Given `EvalSegment.error_type` and agent output:
- `has_issue` correctly True → True Positive
- `has_issue` False when issue exists → False Negative
- Computes per-error-type Precision, Recall, F1
- Reports `critical_error_catch_rate` separately (severity=critical only)
- Reports average F1 across all types

**`cost.py`**  
- `TokenCounter`: wraps LLM calls, accumulates `prompt_tokens` + `completion_tokens`
- `LatencyTimer`: wall-clock per segment (start/stop context manager)
- Outputs per-cell: `{tokens_total, tokens_prompt, tokens_completion, cost_usd, latency_mean_s, latency_p95_s}`

### 5.5 Experiment Runner

**`runner.py`**  
```
for rag_level in [no_rag, vector, hybrid]:
    for coord_level in [single_agent, pipeline, debate]:
        for trial in range(N_TRIALS):
            for segment in eval_segments:
                context = retriever.retrieve(segment.source, segment.draft)
                result  = coordinator.run(segment, context)
                metrics = evaluator.score(segment, result)
                log(cell, trial, segment.error_type, metrics, cost)
```
- `N_TRIALS = 3` (per plan: run-to-run variance reported)
- Fixed temperature = 0 for reproducibility
- Fixed model snapshots (set in `config/models.yaml`)
- Results written to `results/{rag_level}_{coord_level}_trial{n}.jsonl`

---

## 6. Architecture Evaluation — Pass 1: Technical Correctness

**What I checked:**

✅ **RAG and coordination axes are cleanly orthogonal.** The `RAGRetriever` output is a `context: list[TermHit]` passed to the `Coordinator`. Swapping the retriever does not change the coordination logic and vice versa.

✅ **Recall@k proxy is concrete and measurable.** At retrieval time, we know which terms are present in the source. We check whether they appear in the top-k retrieved hits. This is computable without extra annotation.

✅ **Pipeline ≠ single_agent ≠ debate.** The three coordination levels have meaningfully different agent topologies and token costs, making the ordinal scale interpretable.

⚠️ **Issue: "pipeline" is underspecified in the plan.** The plan says `single-agent → pipeline → debate` but doesn't define pipeline's merge logic. I define it as: all 7 specialists run in parallel, suggestions accepted by highest severity, no coordinator call. This needs to be agreed with the supervisor.

⚠️ **Issue: xCOMET requires a GPU or significant RAM.** `xCOMET-XL` needs ~6GB VRAM. On KTH CPU, inference will be slow. Mitigation: batch all segments, use `xCOMET-Lite` for development, `xCOMET-XL` for final results. Or use Unbabel's API.

⚠️ **Issue: sentence_pairs.jsonl has 17,103 entries but the plan cites 45,348.** The full TMX has not been fully processed yet. The evaluation set should be drawn from the processed segments for reproducibility, with a note that the full TMX would strengthen the study.

**Verdict:** Architecture is technically sound. Three issues identified; two are scoping/resource constraints rather than design flaws.

---

## 7. Architecture Evaluation — Pass 2: Research Validity

**Does this answer the RQs?**

**RQ1** (error distribution): The injector creates a balanced per-type set. The evaluation harness computes per-type F1 for each cell. The `no_rag × single_agent` cell (C1) is the baseline distribution. ✅

**RQ2** (interaction effect): Comparing per-type F1 across the grid rows (fixed coordination, varying RAG) shows how RAG grounding changes accuracy per type. The key test: for `terminology` errors, does F1 increase more from `no_rag→hybrid` in `single_agent` than in `debate`? If yes, RAG substitutes for debate on that type. ✅

**RQ3** (cost frontier): Token counts per cell are logged. Plotting F1 vs token cost per cell gives the Pareto frontier. The `no_rag × single_agent` cell is the cheapest; the question is where on the frontier `hybrid × pipeline` sits relative to `no_rag × debate`. ✅

**Threat: synthetic error injection may not match natural error distribution.** Mitigation (from plan): cross-check a 50-segment sample against real translator edits in the Q3/Q1 quarterly segment files. The `splitter.py` enables this.

**Threat: TM repetition confounder.** REIT reports recur heavily across quarters. If TM-matched segments are included, apparent F1 will be inflated because LLMs have likely seen these texts. The `splitter.py` addresses this by separating novel segments. All experiments run on novel segments only; TM-matched segments are reported separately as a secondary analysis.

**Threat: single issuer (Client), single direction (sv→en).** This is acknowledged in the plan as a scope limitation. The architecture does not attempt to generalise; findings are explicitly scoped to this REIT domain.

**Verdict:** The architecture is research-valid for the stated scope. The three threats are acknowledged and partially mitigated within the design.

---

## 8. Architecture Evaluation — Pass 3: Feasibility & Timeline Alignment

**June milestone (Task 1):** Single-agent / no-RAG baseline, error taxonomy, evaluation harness.

What `first_experiment/` must deliver in June:
1. ✅ `data/loader.py` — load existing processed data
2. ✅ `data/splitter.py` — separate TM-match vs novel
3. ✅ `data/injector.py` — 6-type error injection
4. ✅ `coordination/single_agent.py` — one-prompt full review
5. ✅ `eval/metrics.py` — per-error-type F1
6. ⚠️ `eval/xcomet.py` — xCOMET integration: substantial new work; needs GPU or API access confirmed
7. ✅ `rag/no_rag.py` — trivial (no-op)
8. 🔜 `rag/vector_rag.py` / `hybrid_rag.py` — July work; scaffold now, implement later
9. 🔜 `coordination/pipeline.py` / `debate.py` — July work; scaffold now

**Risk: xCOMET setup.** If KTH GPU is unavailable in June, use the COMET library CPU mode or Unbabel API as fallback. The metrics interface is decoupled so the scorer can be swapped.

**Risk: error injector quality.** The injected errors need to look plausible. For terminology injection, use the termbase to find near-misses (e.g., synonyms that are wrong in the IFRS context). This requires a small manual validation pass.

**Scoping decision:** `first_experiment/` implements the **full skeleton** of the grid (all files, all abstractions) but only the **baseline cell** (C1: no_rag × single_agent) runs end-to-end in June. All other cells are scaffolded with `NotImplementedError` stubs, filled in July–September.

**Verdict:** Architecture is feasible for the June milestone if scoped to C1. The vector/hybrid RAG and pipeline/debate coordination are July work. The critical path is xCOMET integration and error injection quality.

---

## 9. Revised Architecture Summary (Post-Evaluation)

Three revisions from the evaluation passes:

1. **Pipeline merge logic explicitly defined** (Pass 1): highest-severity suggestion per dimension, no coordinator call, fully parallel.

2. **Segment stratification mandatory** (Pass 2): `splitter.py` is not optional — it is required for validity. Experiments report novel-only results as the primary analysis.

3. **Phased implementation** (Pass 3): `first_experiment/` ships the full skeleton + baseline (C1) in June. RAG retrieval cells (C2, C3) and debate cells (C7, C8, C9) ship in July–August.

---

## 10. File Structure

```
first_experiment/
├── README.md
├── config/
│   ├── grid.yaml          # 3×3 grid config, N_TRIALS, eval segment count
│   └── models.yaml        # fixed model snapshots + temperature
├── src/
│   ├── data/
│   │   ├── loader.py      # Segment, Term dataclasses + JSONL readers
│   │   ├── injector.py    # 6-type error injection
│   │   └── splitter.py    # TM-match vs novel separation
│   ├── rag/
│   │   ├── base.py        # RAGRetriever ABC + TermHit dataclass
│   │   ├── no_rag.py      # NoRAG: returns []
│   │   ├── vector_rag.py  # VectorRAG: sentence-transformers + FAISS
│   │   └── hybrid_rag.py  # HybridRAG: BM25 + dense re-rank
│   ├── coordination/
│   │   ├── base.py        # Coordinator ABC + ReviewResult dataclass
│   │   ├── single_agent.py # SingleAgent: one full-review call
│   │   ├── pipeline.py    # Pipeline: 7 parallel specialists, rule merge
│   │   └── debate.py      # Debate: CHORUS Round1 + Round2
│   ├── eval/
│   │   ├── xcomet.py      # xCOMET span scorer wrapper
│   │   ├── metrics.py     # per-error-type P/R/F1
│   │   └── cost.py        # TokenCounter + LatencyTimer
│   └── runner.py          # grid experiment runner
├── scripts/
│   ├── run_baseline.py    # June: C1 (no_rag × single_agent)
│   └── run_grid.py        # July+: full 9-cell grid
└── results/
    └── .gitkeep
```

---

## 11. Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| RAG proxy metric | recall@k over termbase union | Measurable at retrieval time; recommended by plan |
| Vector store | FAISS (local) | No external dependency; fast for 3K-term KB |
| BM25 implementation | rank_bm25 | Pure Python; deterministic |
| Embeddings | sentence-transformers/paraphrase-multilingual-mpnet-base-v2 | Handles sv+en bilingual retrieval |
| xCOMET model | xCOMET-XL (GPU) / xCOMET-Lite (CPU fallback) | Span-level scoring as per plan |
| Temperature | 0.0 | Fixed for reproducibility |
| Trial count | 3 | Balances run-to-run variance reporting vs API cost |
| Eval set size | 200 novel segments × 6 error types = 1,200 segment-error pairs | Power consideration per plan |
| Cost reporting | token count (primary), USD (secondary) | Provider-neutral as per plan |
