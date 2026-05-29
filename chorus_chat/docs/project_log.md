# CHORUS Chat Project Log

Last updated: 2026-05-09

## Project Goal

Build a local CHORUS-style multi-agent translation revision MVP for Swedish-to-English financial translation.

The system reviews a draft translation using several specialist agents, then combines their findings through a second-round coordinator. The goal is to support human translators with structured, evidence-based revision suggestions rather than fully automatic replacement.

## Current Scope

- Source language: Swedish
- Target language: English
- Domain: financial translation, with emphasis on IFRS/accounting terminology
- Runtime during testing: local only
- LLM backend: Ollama
- Default model: `qwen3:8b`

The project no longer uses DeepSeek/OpenAI during the test stage.

## Current CHORUS Flow

1. Round 1 specialist agents independently review one translation dimension each.
2. Each agent returns structured JSON with issue status, severity, affected span, suggested revision, explanation, and confidence.
3. Round 2 coordinator compares all agent outputs.
4. Coordinator accepts supported suggestions, rejects out-of-scope or unsupported suggestions, resolves conflicts, and produces one final recommendation.
5. The result is saved to a model-specific JSON file under `outputs/`.

## Round 1 Agents

- `accuracy`: checks whether the English translation preserves the Swedish source meaning.
- `terminology`: checks IFRS/accounting terminology and consistency between Swedish and English.
- `fluency`: checks English grammar, word order, readability, and naturalness.
- `style`: checks formal financial reporting style and register.
- `audience_appropriateness`: checks whether the translation fits professional financial translators, auditors, investors, or report readers.
- `locale_convention`: checks regional conventions, date/number/currency format, and British vs American spelling.
- `design_markup`: checks formatting, tags, placeholders, line breaks, tables, symbols, and structural integrity.

## Main Files

- `src/chorus_mvp/agents.py`: Agent definitions, Round 1 prompt construction, and agent execution.
- `src/chorus_mvp/debate.py`: Round 2 coordinator prompt and consolidation logic.
- `src/chorus_mvp/llm.py`: Local Ollama client utilities and JSON parsing.
- `src/chorus_mvp/run_agents.py`: CLI entry point for running the full CHORUS check.
- `src/test_ollama_qwen_translation.py`: Simple local Ollama + Qwen smoke test.
- `scripts/run_benchmark_v0.py`: Runs the JSONL benchmark across one or more local Ollama models and writes per-case JSON/stdout/stderr files.
- `tests/test_agents.py`: Tests for agent prompts and result normalization.
- `tests/test_debate.py`: Tests for coordinator prompts and result normalization.
- `tests/test_llm.py`: Tests for JSON extraction.

## Current Folder Structure

```text
chorus_chat/
├── docs/
│   └── project_log.md
├── outputs/
│   └── agent_results.json
├── src/
│   ├── chorus_mvp/
│   │   ├── __init__.py
│   │   ├── agents.py
│   │   ├── debate.py
│   │   ├── llm.py
│   │   └── run_agents.py
│   ├── llm_clients/
│   │   └── __init__.py
│   └── test_ollama_qwen_translation.py
├── tests/
│   ├── test_agents.py
│   ├── test_debate.py
│   └── test_llm.py
├── pyproject.toml
└── requirements.txt
```

## Completed Changes

- Changed default source language to Swedish.
- Changed default target language to English.
- Changed default example source/draft to Swedish-to-English.
- Switched test-stage runtime to local Ollama + Qwen.
- Set default model to `qwen3:8b`.
- Added `CHORUS_OLLAMA_MODEL` environment override.
- Removed DeepSeek/OpenAI client and DeepSeek test script.
- Added a local Qwen translation smoke test.
- Reworked Round 1 agent prompts to better match CHORUS:
  - independent specialist review
  - dimension isolation
  - evidence grounded in source/draft
  - minimal suggested corrections
  - severity calibration
- Updated the Round 1 agent structure to the CHORUS-style dimensions:
  - Accuracy
  - Terminology
  - Fluency
  - Style
  - Audience Appropriateness
  - Locale Convention
  - Design and Markup
- Reworked Round 2 coordinator prompt to better match CHORUS:
  - compare evidence
  - reject unsupported or out-of-scope suggestions
  - resolve conflicts
  - preserve draft when no valid issue exists
- Added basic unit tests.
- Changed default output behavior so each model writes to a separate JSON file.
- Fixed the benchmark runner so each case/model writes directly to its own JSON output path.

## How To Run

Check installed Ollama models:

```bash
ollama list
```

Run the full CHORUS pipeline with default sample text:

```bash
cd /home/cmy_rick/projects/thesis/chorus_chat
env PYTHONPATH=src python3 src/chorus_mvp/run_agents.py
```

By default, output is grouped by model:

```text
outputs/agent_results_qwen3_8b.json
outputs/agent_results_llama3.1_8b.json
```

You can still override the output path manually:

```bash
env PYTHONPATH=src python3 src/chorus_mvp/run_agents.py \
  --model llama3.1:8b \
  --output outputs/custom_llama_run.json
```

Run with custom Swedish source and English draft:

```bash
env PYTHONPATH=src python3 src/chorus_mvp/run_agents.py \
  --source "Jag skulle vilja boka ett möte nästa vecka." \
  --draft "I would like book a meeting next week."
```

Run the local Qwen smoke test:

```bash
env PYTHONPATH=src python3 src/test_ollama_qwen_translation.py
```

Run unit tests:

```bash
env PYTHONPATH=src:. python3 -m unittest discover -s tests
```

Run the benchmark:

```bash
env PYTHONPATH=src:. python3 scripts/run_benchmark_v0.py \
  --benchmark data/benchmark/financial_qa_benchmark_v0.jsonl \
  --models qwen3:8b,llama3.1:8b \
  --include-domain
```

## Latest Verification

Unit tests passed:

```text
Ran 13 tests
OK
```

Local Qwen smoke test returned:

```text
I would like to book a meeting next week.
```

## Known Cleanup Items

- `src/llm_clients/__init__.py` is now unused after removing the DeepSeek client.
- `outputs/agent_results.json` is generated output and may not need to be tracked.
- `.venv/` is local environment state and should not be tracked.
- Add a proper `README.md`.
- Consider adding a `.gitignore`.

## Suggested Next Steps

- Add `README.md` with setup, Ollama requirements, and example commands.
- Add `.gitignore` for `.venv/`, `__pycache__/`, `.env`, and generated outputs.
- Decide whether to keep or remove `src/llm_clients/`.
- Add a small dataset-driven evaluation script for Swedish-to-English financial examples.
- Add stronger schema validation for LLM JSON responses.
- Add optional batch mode for multiple translation segments.
- Add memory/context handling if translator behavior patterns should be reused across segments.
