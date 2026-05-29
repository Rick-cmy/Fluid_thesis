# first_experiment — Financial Translation QA: RAG vs. Multi-Agent Coordination

**Research question:** In Swedish–English financial translation, which error types can be fixed by retrieving the right terminology (RAG), and which require multiple agents working together? And when RAG is strong enough, does adding more agents still help?

This is the controlled experiment system behind Mingyang Chen's KTH master's thesis, conducted in collaboration with Fluid Translation.

---

## What This System Does

A translation QA system reviews a Swedish→English financial document segment and flags errors across seven quality dimensions (accuracy, terminology, fluency, style, locale conventions, audience fit, markup). The system can be configured along two axes:

**Axis 1 — How much retrieval context the agents get:**
- `no_rag` — agents see only the source and draft translation
- `vector` — agents receive the top-5 relevant IFRS/Client terminology matches retrieved from a 3,266-term database
- `hybrid` — same as vector but combined with BM25 keyword matching for higher recall

**Axis 2 — How many agents collaborate:**
- `single_agent` — one LLM call reviews all dimensions at once
- `pipeline` — seven specialist agents each review one dimension independently, in parallel; results are merged automatically
- `debate` — same seven agents in Round 1, then a coordinator agent reads all their outputs, resolves conflicts, and writes one final recommendation (the CHORUS protocol)

This gives a **3×3 grid of nine configurations**. The experiment runs all nine, measuring accuracy, latency, and token cost per error type, to find which combination is best for which kinds of errors.

---

## Agent Structure

### The Seven Specialist Agents (Round 1)

Each agent is a language model prompted to review **one dimension only**. They run in parallel.

| Agent | Dimension | What it checks |
|-------|-----------|----------------|
| `accuracy` | Accuracy | Meaning preservation — omissions, additions, distortions |
| `terminology` | Terminology | IFRS/IAS domain terms, Client-specific glossary consistency |
| `fluency` | Fluency | Grammar, naturalness, readability in English |
| `style` | Style | Tone, register, formality appropriate for financial reports |
| `locale_convention` | Locale | Date/number/currency format; British vs. American spelling |
| `audience_appropriateness` | Audience | Fit for professional translators, auditors, and investors |
| `design_markup` | Markup | Formatting, tags, placeholders, table structure |

Each agent returns structured JSON: `{has_issue, severity, issue_span, suggested_revision, explanation}`.

Severity levels follow MQM: `none → minor → major → critical`.

### How Agents Collaborate (the CHORUS Protocol)

```
Source text + Draft translation
          │
          ├─ [RAG retriever] ──► terminology context (optional)
          │
          ▼
┌─────────────────────────────────────────────────────┐
│  ROUND 1  (all 7 agents run simultaneously)         │
│                                                     │
│  accuracy ──┐                                       │
│  terminology─┤                                      │
│  fluency ───┤  each reviews its own dimension       │
│  style ─────┤  independently, returns JSON          │
│  locale ────┤                                       │
│  audience ──┤                                       │
│  markup ────┘                                       │
└─────────────────────────────────────────────────────┘
          │
          │  (pipeline stops here — rule-based merge)
          │  (debate continues ↓)
          ▼
┌─────────────────────────────────────────────────────┐
│  ROUND 2  Coordinator agent                         │
│                                                     │
│  - Reads all 7 Round 1 outputs                      │
│  - Accepts suggestions supported by the source text │
│  - Rejects out-of-scope or unsupported claims       │
│  - Resolves conflicts between agents                │
│  - Writes one final revised translation             │
└─────────────────────────────────────────────────────┘
          │
          ▼
  Final recommendation + reasoning summary
  (human translator can accept, edit, or reject)
```

**Pipeline** stops after Round 1 and picks the highest-severity suggestion automatically — no cross-agent communication. It is faster and cheaper than Debate, and the experiment measures whether the quality gap is worth the extra cost.

---

## Error Types Under Study

Errors are pre-classified into two groups, which is the core of the research hypothesis:

**Retrieval-addressable** — errors that can in principle be fixed by providing the right terminology at retrieval time, without needing agents to discuss with each other:

| Type | Example |
|------|---------|
| `terminology` | "depreciation" written instead of "impairment loss" |
| `numeracy` | "SEK 5.2 thousand" instead of "SEK 5.2 million" |
| `named_entity` | Wrong company name or ISIN code |

**Coordination-addressable** — errors that require judgment, context, or knowledge of Client's house style that no terminology database can supply:

| Type | Example |
|------|---------|
| `fluency` | Grammatically awkward phrasing |
| `style_guide` | "percent" instead of "per cent"; wrong date format |
| `consistency` | "profit increased" when source says profit fell |

The hypothesis: as RAG grounding gets stronger, the extra accuracy from adding more agents shrinks for retrieval-addressable errors — but not for coordination-addressable ones.

---

## Current Results

See [`../PROGRESS_LOG.md`](../PROGRESS_LOG.md) for all experimental results and per-run notes.

---

## Running Locally

### Requirements

- Python 3.10+
- [Ollama](https://ollama.com) installed and running (`ollama serve`)
- At least one model pulled:

```bash
ollama pull qwen3:8b      # best quality
ollama pull llama3.2:1b   # fastest for quick tests
```

- Python dependencies:

```bash
pip install pyyaml httpx
```

For cloud API runs, set the relevant API key (provider TBD):
```bash
export DEEPSEEK_API_KEY=sk-...   # example — replace with chosen provider's key
```

### Quick smoke test — verify everything works (≈ 5–10 minutes on CPU)

```bash
cd first_experiment
python scripts/run_baseline.py --model qwen3:8b --n-per-type 5
```

Results appear in `results/baseline/baseline_summary.json`.

### Test all three coordinator types in parallel (SingleAgent / Pipeline / Debate)

```bash
python scripts/smoke_test_parallel.py --model qwen3:8b
```

Runs 3 segments through each coordinator. Verifies that the 7-agent parallel execution works and shows comparative latency.

### Run the baseline properly (200 segments per error type)

```bash
python scripts/run_baseline.py --model qwen3:8b --n-per-type 200
```

Recommended: use a cloud LLM API to avoid multi-hour CPU runs (provider TBD):
```bash
python scripts/run_baseline.py --model <model-name> --provider <provider> --n-per-type 200
```

### Run the full 3×3 grid

```bash
python scripts/run_grid.py --model <model-name> --provider <provider> --n-per-type 200 --trials 3
```

Grid cells requiring VectorRAG or HybridRAG (not yet implemented) are skipped automatically. Results go to `results/grid/`.

---

## Data

The experiment uses three datasets provided by Fluid Translation:

| Dataset | Description | Size |
|---------|-------------|------|
| Master TMX | Human-translated sv→en-GB Client sentence pairs | 17,103 segments |
| Client termbase | Client-specific Swedish–English terminology | 1,938 terms |
| Quarterly reports | Q3 2025 + Q1 2026 segment files | 5,296 segments |

**Data handling:**
1. Segments that appear verbatim in the quarterly reports are removed from the evaluation pool (to avoid inflating scores with translation-memory exact matches).
2. The remaining 16,834 novel segments are split 80/20 into a **test set** (used only for final grid experiments) and a **dev set** (used for development and debugging).
3. Controlled errors are injected into segments to create a balanced evaluation set — one error type per segment, with the correct translation known.

---

## Project Structure

```
first_experiment/
├── config/
│   ├── grid.yaml          # grid config: RAG levels, coordination levels, split ratios
│   └── models.yaml        # LLM model snapshots and provider settings
├── src/
│   ├── llm.py             # unified LLM client (Ollama + external API)
│   ├── data/
│   │   ├── loader.py      # reads JSONL data files
│   │   ├── splitter.py    # TM-match removal + dev/test split
│   │   └── injector.py    # controlled error injection (6 types)
│   ├── rag/
│   │   ├── no_rag.py      # no retrieval (baseline)
│   │   ├── vector_rag.py  # dense retrieval (in progress)
│   │   └── hybrid_rag.py  # BM25 + dense (in progress)
│   ├── coordination/
│   │   ├── single_agent.py  # one prompt, all dimensions
│   │   ├── pipeline.py      # 7 parallel agents, rule-based merge
│   │   └── debate.py        # 7 parallel agents + coordinator (CHORUS)
│   └── eval/
│       ├── metrics.py     # per-error-type Precision / Recall / F1
│       ├── cost.py        # token count + latency tracking
│       └── xcomet.py      # xCOMET span scorer (in progress)
├── scripts/
│   ├── run_baseline.py          # run C1: no_rag × single_agent
│   ├── smoke_test_parallel.py   # test all coordinators on 3 segments
│   └── run_grid.py              # run full 3×3 grid
└── results/                     # output JSONL files per grid cell
```

---

## References

- **CHORUS** (Wang et al., 2026) — the multi-agent MQM review protocol this system implements
- **MAATS** (Wang et al., 2025) — MQM-based multi-agent translation system
- **xCOMET** (Guerreiro et al., 2024) — span-level translation quality metric used for scoring
- **MAST** (Cemri et al., 2025) — taxonomy of multi-agent system failures; informs the coordinator design
- Full literature: `../papers/` (76 PDFs)
