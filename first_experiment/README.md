# first_experiment — Financial Translation QA: RAG × Multi-Agent Coordination

**Research question:** In Swedish–English financial translation, which error types can be fixed by retrieving the right terminology or style rules (RAG), and which require multiple agents to coordinate? And when RAG is strong enough, does adding more agents still help?

This is the controlled 3×3 grid experiment behind Mingyang Chen's KTH master's thesis, conducted in collaboration with Fluid Translation.

---

## What This System Does

A translation QA system reviews a Swedish→English financial document segment and flags errors across six quality dimensions. The system is configured along two independent axes:

**Axis 1 — Retrieval context (RAG level):**
- `no_rag` — agents see only the source and draft translation
- `term_rag` — agents receive the top-5 IFRS/Client terminology matches retrieved via FAISS dense retrieval from the 1,938-term Client termbase
- `rich_rag` — same as `term_rag`, plus the top-3 Client Financial Style Guide rules retrieved from a separate FAISS index over 27 machine-readable rules

**Axis 2 — Coordination strategy:**
- `single_agent` — one LLM call reviews all six dimensions simultaneously
- `pipeline` — six specialist agents each review one dimension independently in parallel; results are merged by highest severity (no cross-agent communication)
- `debate` — six specialist agents run in parallel (Phase 1), then a single meta-coordinator synthesises their findings using confidence thresholding and cross-agent corroboration (Phase 2) — the **CHORUS-v2** protocol

This gives a **3×3 grid of nine configurations**. Each cell is evaluated on six error types with full precision / recall / F1 and token cost tracking.

---

## The Six Specialist Agents

Each agent reviews **one dimension only**. They run in parallel (Pipeline Phase 1 and Debate Phase 1 are identical).

| Agent | Dimension | What it checks |
|-------|-----------|----------------|
| `terminology` | Terminology | IFRS/IAS financial terms and Client glossary consistency (e.g. "income" vs "revenue", "fair value" vs "market value") |
| `numeracy` | Numeracy | Numbers, amounts, percentages, dates, currencies — every figure must match the source exactly |
| `named_entity` | Named entity | Company names, property names, accounting standards (IFRS/IAS/GAAP), index names, geographic names |
| `fluency` | Fluency | Grammar, naturalness, sentence structure, readability in English |
| `style_guide` | Style | Client Financial Style Guide: decimal separator (period not comma), "percent" not "per cent", DD Month YYYY date format, British English spelling |
| `consistency` | Consistency | Meaning consistency with source — flags directional/polarity errors (increased/decreased, profit/loss) |

Each agent returns structured JSON:
```json
{
  "agent": "terminology",
  "has_issue": true,
  "severity": "none|minor|major|critical",
  "issue_span": "problematic part of the draft",
  "suggested_revision": "corrected phrase",
  "explanation": "brief explanation",
  "confidence": 0.85
}
```

---

## Debate: CHORUS-v2 Protocol

The `debate` coordinator implements CHORUS-v2, a 2-phase meta-coordination protocol:

```
Source text + Draft translation
          │
          ├─ [RAG retriever] ──► terminology + style-rule context (optional)
          │
          ▼
┌─────────────────────────────────────────────────────────┐
│  PHASE 1  (6 agents run simultaneously, 6 LLM calls)    │
│                                                         │
│  terminology ──┐                                        │
│  numeracy ─────┤  each reviews its own dimension        │
│  named_entity ─┤  independently, returns JSON           │
│  fluency ──────┤  with confidence score [0, 1]          │
│  style_guide ──┤                                        │
│  consistency ──┘                                        │
└─────────────────────────────────────────────────────────┘
          │
          │  (pipeline stops here — rule-based merge)
          │  (debate continues ↓)
          ▼
┌─────────────────────────────────────────────────────────┐
│  PHASE 2  Meta-coordinator (1 LLM call)                 │
│                                                         │
│  - Groups findings: with_issues first, then no_issues   │
│  - ACCEPTS: has_issue=true AND confidence ≥ 0.65        │
│             OR severity="critical" (regardless)         │
│  - REJECTS: confidence < 0.40 AND not critical          │
│  - CROSS-VALIDATES: overlapping spans across agents     │
│  - RESOLVES: conflicts between agents                   │
│  - PRODUCES: one complete revised translation           │
└─────────────────────────────────────────────────────────┘
          │
          ▼
  Final recommendation + accepted/rejected points + reasoning summary
```

**Token cost:** Pipeline = 6 calls/segment. Debate (CHORUS-v2) = 7 calls/segment.

Design references: ChatEval (Chan et al., 2024) simultaneous-talk-with-summarizer; ReConcile (Chen et al., 2024) confidence scoring and grouped presentation; ManyMinds (Ma et al., 2025) meta-judge avoids bias amplification from iterative debate.

---

## Error Types Under Study

| Type | Category | Hypothesis |
|------|----------|-----------|
| `terminology` | retrieval-addressable | Stronger RAG → smaller marginal gain from coordination |
| `numeracy` | retrieval-addressable | Same |
| `named_entity` | retrieval-addressable | Same |
| `fluency` | coordination-addressable | Coordination necessary regardless of RAG |
| `consistency` | coordination-addressable | Same |
| `style_guide` | **borderline (pivot)** | Retrieval-addressable only under `rich_rag` (style rules retrieved explicitly) |

The core measurement: `ΔRecall_coord(rag) = Recall(pipeline, rag) − Recall(single_agent, rag)` — this delta should decrease as `rag_level` increases for retrieval-addressable types, but stay flat for coordination-addressable types.

---

## Current Results (no_rag cells)

| Cell | macro F1 | terminology | numeracy | named_entity | fluency | style_guide | consistency |
|------|----------|-------------|----------|--------------|---------|-------------|-------------|
| no_rag × single_agent | 0.740 | 0.907 | 0.952 | 0.919 | 0.347 | 0.326 | 0.990 |
| no_rag × pipeline | 0.962 | — | — | — | — | — | — |
| no_rag × debate | 0.961 | — | — | — | — | — | — |

Full grid results pending (term_rag and rich_rag cells currently running).

---

## Running Locally

### Requirements

- Python 3.10+
- [Ollama](https://ollama.com) installed and running (`ollama serve`)

```bash
ollama pull qwen3:8b
```

- Python dependencies (use venv — Ubuntu 24.04 requires it):

```bash
cd first_experiment
python3 -m venv .venv && source .venv/bin/activate
pip install pyyaml sentence-transformers faiss-cpu
```

### Smoke test (≈ 5 minutes)

```bash
python scripts/run_baseline.py --model qwen3:8b --n-per-type 5
```

### Full 3×3 grid (n=200/type, ~183h on local GPU)

```bash
python scripts/run_grid.py --provider ollama --model qwen3:8b --n-per-type 200 --trials 1
```

The runner checkpoints each segment to `results/grid/*.jsonl` — safe to interrupt and resume with the same command.

---

## Data

| Dataset | Description | Size |
|---------|-------------|------|
| Master TMX | Human-translated sv→en-GB Client sentence pairs | 17,103 segments |
| Client termbase | Client-specific Swedish–English terminology | 1,938 terms |
| Quarterly reports | Q3 2025 + Q1 2026 segment files | 5,296 segments |

Data handling:
1. Segments appearing verbatim in the quarterly reports are removed from the evaluation pool (TM-match filter).
2. The remaining 16,834 novel segments are split 80/20 → **test set** (grid evaluation) / **dev set** (development).
3. Controlled errors are injected per segment (one error type per segment, known ground truth) to create 200 × 6 = 1,200 positive segments plus 200 clean negatives per cell.

---

## Project Structure

```
first_experiment/
├── config/
│   ├── grid.yaml          # rag_levels, coordination_levels, split config, eval counts
│   └── models.yaml        # LLM model snapshots and provider settings
├── src/
│   ├── llm.py             # unified LLM client (Ollama + DeepSeek API)
│   ├── runner.py          # run_cell + run_grid; checkpoint/resume logic
│   ├── data/
│   │   ├── loader.py      # reads JSONL data files
│   │   ├── splitter.py    # TM-match removal + dev/test split
│   │   └── injector.py    # controlled error injection (6 types) + clean set
│   ├── rag/
│   │   ├── base.py        # RAGRetriever ABC
│   │   ├── no_rag.py      # no retrieval (baseline)
│   │   ├── term_rag.py    # FAISS dense retrieval over Client termbase
│   │   └── rich_rag.py    # term_rag + FAISS over style rules
│   ├── coordination/
│   │   ├── base.py        # Coordinator ABC, DimensionResult, ReviewResult dataclasses
│   │   ├── single_agent.py  # one prompt, all 6 dimensions
│   │   ├── pipeline.py      # 6 parallel specialists, rule-based merge
│   │   └── debate.py        # CHORUS-v2: 6 parallel + 1 meta-coordinator (7 calls)
│   └── eval/
│       ├── metrics.py     # per-error-type Precision / Recall / F1; macro F1
│       └── cost.py        # token count + latency tracking
├── scripts/
│   ├── run_baseline.py          # run no_rag × single_agent
│   ├── run_grid.py              # run full 3×3 grid
│   ├── smoke_test_parallel.py   # test all coordinators on 3 segments
│   └── analyze_grid.py          # bootstrap CI, McNemar's, interaction effects
└── results/                     # JSONL checkpoints per grid cell (gitignored)
```

---

## References

- **CHORUS-v2** — this system's debate protocol; grounded in ChatEval (Chan et al., 2024), ReConcile (Chen et al., 2024), ManyMinds (Ma et al., 2025)
- **GEMBA-MQM** (Kocmi & Federmann, 2023) — LLM-as-judge for MQM error span detection
- **MAATS** (Wang et al., 2025) — MQM-based multi-agent translation evaluation system
- **MAST** (Cemri et al., 2025) — taxonomy of multi-agent system failures
- Full literature: 130 papers in `../papers/` (gitignored)
