# Retrieval or Coordination?

**An Error-Type Analysis of When RAG Terminology Grounding Can Reduce Multi-Agent Coordination in Financial Translation QA**
*(Swedish–English REIT Case Study)*

KTH Royal Institute of Technology — Master's Thesis, 2026
Mingyang Chen

---

## Research Question

In Swedish–English financial translation QA, can stronger retrieval (RAG) substitute for multi-agent coordination — and if so, for which error types?

**Hypothesis:** For retrieval-addressable errors (terminology, numeracy, named entity), stronger RAG reduces the marginal gain of adding coordination. For coordination-addressable errors (fluency, consistency), coordination remains necessary regardless of RAG strength. `style_guide` is the pivot case: only becomes retrieval-addressable under `rich_rag`.

---

## Repository Structure

```
├── first_experiment/       # Controlled 3×3 grid study (main experiment)
├── style_classifier/       # Binary Transformer classifier for style-guide violations
├── chorus_chat/            # Original CHORUS prototype (reference implementation)
├── transagents_baseline_chat/  # Single-agent Ollama baseline
├── fluid_qa/               # Productized QA engine for Fluid Translation (memoQ integration)
├── agents_architecture.md  # Full system architecture
├── scripts/
│   └── download.py         # Literature download script (arXiv + ACL Anthology)
└── data/                   # Gitignored — confidential client data
```

The main experiment lives in [`first_experiment/`](first_experiment/README.md). It implements a **3×3 grid**:

|  | `single_agent` | `pipeline` | `debate` |
|---|---|---|---|
| `no_rag` | ✅ done | ✅ done | ✅ done |
| `term_rag` | 🔄 running | ⏳ pending | ⏳ pending |
| `rich_rag` | ⏳ pending | ⏳ pending | ⏳ pending |

Each cell is evaluated on six error types (precision / recall / F1) with token cost and latency tracking.

---

## Data

Provided by [Fluid Translation](https://www.fluidtranslation.se/) under a research agreement — not included in this repository.

| Dataset | Description | Size |
|---------|-------------|------|
| Client TMX | Human-translated sv→en-GB sentence pairs | 17,103 segments |
| Client termbase | Client-specific Swedish–English terminology | 1,938 terms |
| Quarterly reports | Client Q3 2025 + Q1 2026 segment files | 5,296 segments |
| Style rules | Client Financial Style Guide rules (machine-readable) | 27 rules |

---

## Quick Start

```bash
# Create and activate venv (required — externally-managed Python)
cd first_experiment
python3 -m venv .venv && source .venv/bin/activate

# Install dependencies
pip install pyyaml sentence-transformers faiss-cpu

# Run baseline smoke test (requires Ollama + qwen3:8b)
ollama pull qwen3:8b
python scripts/run_baseline.py --model qwen3:8b --n-per-type 5
```

See [`first_experiment/README.md`](first_experiment/README.md) for full usage.

---

## Status

- [x] Experiment framework complete (3 coordinators × 3 RAG levels, 6 error types, metrics, cost tracking)
- [x] `no_rag × single_agent`: macro F1 = 0.740 (fluency/style lowest as hypothesised)
- [x] `no_rag × pipeline`: macro F1 = 0.962
- [x] `no_rag × debate` (CHORUS-v2): macro F1 = 0.961
- [x] `term_rag` and `rich_rag` retrievers implemented (FAISS dense retrieval)
- [x] `debate.py` redesigned as CHORUS-v2 (7 calls: 6 parallel specialists + 1 meta-coordinator)
- [x] `style_classifier` complete: val macro F1 = 0.9618, adversarial defense complete
- [ ] Full-scale grid run in progress (n=200/type, qwen3:8b via local Ollama)
- [ ] Analysis scripts (`analyze_grid.py`): bootstrap CI, McNemar's test, interaction effect
