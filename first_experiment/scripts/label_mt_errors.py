"""
label_mt_errors.py — Classify MT error type for each (mt_en, gold_en) pair.

Model: see config/models.yaml [annotation] — default qwen3:8b via Ollama.

Reads:  data/mt_eval/aligned_*.jsonl
Writes: data/mt_eval/labelled_*.jsonl

Each output line adds:
  error_type   : one of the 6 canonical types (or "none" if no real error)
  error_span   : short quote from mt_en that contains the error
  explanation  : one sentence

Run:
  python scripts/label_mt_errors.py [--limit N] [--model MODEL]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

THESIS_ROOT = Path(__file__).resolve().parents[2]
MT_DIR = THESIS_ROOT / "data" / "mt_eval"
CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "models.yaml"

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from llm import call_llm  # noqa: E402

INPUTS = [
    ("aligned_q3_2025.jsonl", "labelled_q3_2025.jsonl"),
    ("aligned_q1_2026.jsonl", "labelled_q1_2026.jsonl"),
]

ERROR_TYPES = [
    "terminology",   # wrong IFRS/domain term (e.g. "income" vs "revenue")
    "numeracy",      # wrong number, magnitude, date, currency
    "named_entity",  # wrong company, city, standard, index name
    "fluency",       # grammatical error, missing word, broken syntax
    "style_guide",   # Client style rule violated (spelling, date format, etc.)
    "consistency",   # contradicts source meaning (increased/decreased, profit/loss)
    "none",          # MT and gold differ only in CAT-tool tags or whitespace
]

_SYSTEM = """\
You are a financial translation QA specialist for Swedish→English IFRS reports.
You will be given a Swedish source segment, an MT draft, and the human gold translation.
Identify the single most important error type in the MT draft compared to the gold.

Error types:
- terminology   : wrong IFRS/domain term (e.g. "income" vs "revenue")
- numeracy      : wrong number, magnitude, date, or currency
- named_entity  : wrong company, city, standard, or index name
- fluency       : grammatical error, missing word, or broken syntax
- style_guide   : Client style rule violated (e.g. "percent" vs "per cent", date format)
- consistency   : MT contradicts source meaning (e.g. increased vs decreased)
- none          : MT and gold differ only in inline CAT-tool tags or whitespace (no real error)

Respond with JSON only:
{
  "error_type": "<one of the 7 values above>",
  "error_span": "<short verbatim quote from mt_en that contains the error, or empty string if none>",
  "explanation": "<one sentence>"
}"""


def classify_batch(rows: list[dict], model: str, provider: str, base_url: str) -> list[dict]:
    out = []
    for row in rows:
        user_msg = (
            f"SOURCE (sv): {row['source_sv']}\n"
            f"MT DRAFT:    {row['mt_en']}\n"
            f"GOLD:        {row['gold_en']}"
        )
        messages = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user_msg},
        ]
        result, _, _ = call_llm(messages, model=model, provider=provider, base_url=base_url)
        error_type = result.get("error_type", "none") if isinstance(result, dict) else "none"
        if error_type not in ERROR_TYPES:
            error_type = "none"
        out.append({
            "error_type": error_type,
            "error_span": result.get("error_span", "") if isinstance(result, dict) else "",
            "explanation": result.get("explanation", "") if isinstance(result, dict) else "",
        })
    return out


def load_config() -> dict:
    with CONFIG_PATH.open() as f:
        cfg = yaml.safe_load(f)
    return cfg.get("annotation", {})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Max pairs per file (0=all)")
    parser.add_argument("--model", type=str, default="", help="Override model_id from config")
    parser.add_argument("--batch-size", type=int, default=0, help="Pairs per LLM call (0=from config)")
    parser.add_argument("--provider", type=str, default="", choices=["", "ollama", "deepseek"],
                        help="Override provider from config (ollama|deepseek)")
    parser.add_argument("--out-suffix", type=str, default="",
                        help="Suffix appended to output filenames, e.g. '_deepseek' → labelled_q3_2025_deepseek.jsonl")
    args = parser.parse_args()

    cfg = load_config()
    model = args.model or cfg.get("model_id", "qwen3:8b")
    provider = args.provider or cfg.get("provider", "ollama")
    base_url = cfg.get("base_url", "http://127.0.0.1:11434")
    batch_size = args.batch_size or cfg.get("batch_size", 10)

    print(f"Model: {model} ({provider})  batch_size={batch_size}")
    print(f"Config: {CONFIG_PATH}\n")

    for in_file, out_file in INPUTS:
        in_path = MT_DIR / in_file
        if args.out_suffix:
            stem, ext = out_file.rsplit(".", 1)
            out_file = f"{stem}{args.out_suffix}.{ext}"
        out_path = MT_DIR / out_file

        if not in_path.exists():
            print(f"SKIP (missing): {in_path}")
            continue

        rows = []
        with in_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))

        if args.limit:
            rows = rows[: args.limit]

        # Resume: skip already-labelled rows
        done_uuids: set[str] = set()
        if out_path.exists():
            with out_path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        done_uuids.add(json.loads(line).get("uuid", ""))
            rows = [r for r in rows if r.get("uuid") not in done_uuids]

        print(f"{in_file}: {len(rows)} pairs to label (batch_size={batch_size})")
        if not rows:
            print("  All done.\n")
            continue

        counts: dict[str, int] = {t: 0 for t in ERROR_TYPES}
        errors = 0
        labelled = 0
        t0 = time.time()

        with out_path.open("a", encoding="utf-8") as out_f:
            for batch_start in range(0, len(rows), batch_size):
                batch = rows[batch_start : batch_start + batch_size]
                try:
                    labels = classify_batch(batch, model=model, provider=provider, base_url=base_url)
                except Exception as e:
                    print(f"  [batch {batch_start}] ERROR: {e}")
                    errors += len(batch)
                    continue

                for row, label in zip(batch, labels):
                    counts[label["error_type"]] = counts.get(label["error_type"], 0) + 1
                    out_f.write(json.dumps({**row, **label}, ensure_ascii=False) + "\n")
                    labelled += 1

                if labelled % 100 == 0 or batch_start + batch_size >= len(rows):
                    elapsed = time.time() - t0
                    rate = labelled / elapsed if elapsed > 0 else 0
                    remaining = (len(rows) - labelled) / rate if rate > 0 else 0
                    print(f"  {labelled}/{len(rows)}  ({rate:.1f}/s, ~{remaining/60:.0f}m left)  dist={dict(counts)}")

        elapsed = time.time() - t0
        print(f"  Done: {labelled} labelled, {errors} errors, {elapsed:.0f}s")
        print(f"  Distribution: {dict(counts)}")
        print(f"  Written → {out_path}  ({labelled} records)\n")


if __name__ == "__main__":
    main()
