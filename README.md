# Retrieval or Coordination?

**An Error-Type Analysis of When RAG Terminology Grounding Can Reduce Multi-Agent Coordination in Financial Translation QA**
*(Swedish–English REIT Case Study)*

KTH Royal Institute of Technology — Master's Thesis, 2026
Mingyang Chen · Supervisors: Jörg Conradt (KTH), Arvind Kumar (examiner), Gabriella Rudström (Fluid Translation)

---

## Research Question

In Swedish–English financial translation QA, can stronger retrieval (RAG) substitute for multi-agent coordination — and if so, for which error types?

**Hypothesis:** For retrieval-addressable errors (terminology, numeracy, named entity), stronger RAG reduces the marginal gain of adding coordination. For coordination-addressable errors (fluency, style-guide, consistency), coordination remains necessary regardless of RAG strength.

---

## Repository Structure

```
├── first_experiment/       # Controlled 3×3 grid study (main experiment)
├── chorus_chat/            # 7-agent CHORUS prototype (MQM, Swedish→English IFRS)
├── transagents_baseline_chat/  # Single-agent Ollama baseline
├── agents_architecture.md  # Full system architecture and 3-pass evaluation design
├── download.py             # Literature download script (arXiv + ACL Anthology)
└── PROGRESS_LOG.md         # Experiment log
```

The main experiment lives in [`first_experiment/`](first_experiment/README.md). It implements a **3×3 grid**:

| | `single_agent` | `pipeline` | `debate` |
|---|---|---|---|
| `no_rag` | C1 ← baseline | C2 | C3 |
| `vector_rag` | C4 | C5 | C6 |
| `hybrid_rag` | C7 | C8 | C9 |

Each cell is evaluated on six error types (precision / recall / F1) with token cost and latency tracking.

---

## Data

Provided by [Fluid Translation](https://www.fluidtranslation.se/) under a research agreement — not included in this repository.

| Dataset | Description | Size |
|---------|-------------|------|
| Client TMX | Human-translated sv→en-GB sentence pairs | 17,103 segments |
| Client termbase | Client-specific Swedish–English terminology | 1,938 terms |
| Quarterly reports | Client Q3 2025 + Q1 2026 segment files | 5,296 segments |

---

## Quick Start

```bash
# Install dependencies
cd first_experiment
python3 -m venv .venv && source .venv/bin/activate
pip install httpx pyyaml

# Run baseline smoke test (requires Ollama + qwen3:8b)
python scripts/run_baseline.py --model qwen3:8b --n-per-type 5
```

See [`first_experiment/README.md`](first_experiment/README.md) for full usage.

---

## Status

- [x] Experiment framework complete (all 3 coordinators, 6 error types, metrics, cost tracking)
- [x] Baseline (C1) verified: macro F1 = 0.635, fluency/style recall lower as hypothesised
- [ ] VectorRAG implementation
- [ ] Full-scale grid run (n=200/type, cloud API)
- [ ] xCOMET span scoring
