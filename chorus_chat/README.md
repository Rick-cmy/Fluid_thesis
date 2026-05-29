# CHORUS Financial QA Benchmark v0

This is a small benchmark for comparing local LLMs on CHORUS-style financial translation QA.

## Files

- `data/benchmark/financial_qa_benchmark_v0.jsonl`  
  20 Swedish→English financial translation QA cases.

- `scripts/run_benchmark_v0.py`  
  Batch runner for `src/chorus_mvp/run_agents.py`.

- `outputs/manual_eval_template.csv`  
  Manual scoring sheet.

## Install into your project

From inside your project root:

```bash
cd ~/projects/thesis/chorus_chat
mkdir -p data/benchmark scripts
cp /path/to/financial_qa_benchmark_v0.jsonl data/benchmark/
cp /path/to/run_benchmark_v0.py scripts/
```

Or unzip this package into your project root.

## Run

```bash
cd ~/projects/thesis/chorus_chat

python3 scripts/run_benchmark_v0.py \
  --benchmark data/benchmark/financial_qa_benchmark_v0.jsonl \
  --models qwen3:8b,llama3.1:8b
```

If your `run_agents.py` supports `--domain`, use:

```bash
python3 scripts/run_benchmark_v0.py \
  --benchmark data/benchmark/financial_qa_benchmark_v0.jsonl \
  --models qwen3:8b,llama3.1:8b \
  --include-domain
```

## Manual scoring

Use 0/1/2 for each dimension:

- 0 = bad / missed / wrong
- 1 = partially correct
- 2 = correct and useful

Recommended dimensions:

1. detected_core_issue_0_2
2. severity_reasonable_0_2
3. suggestion_quality_0_2
4. explanation_quality_0_2
5. false_positive_penalty_0_2
6. final_revision_quality_0_2

Maximum: 12 points per case.
