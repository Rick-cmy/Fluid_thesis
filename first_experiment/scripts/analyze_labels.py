"""Quick quality check for label_mt annotation results."""

import json
import random
from collections import Counter
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "mt_eval"
FILES = {
    "q3_2025": DATA_DIR / "labelled_q3_2025.jsonl",
    "q1_2026": DATA_DIR / "labelled_q1_2026.jsonl",
}
VALID_TYPES = {"terminology", "numeracy", "named_entity", "fluency", "style_guide", "consistency", "none"}
RANDOM_SEED = 42


def load_jsonl(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  [WARN] line {i}: {e}")
    return records


def check_file(name, records):
    print(f"\n{'='*60}")
    print(f"  {name}  ({len(records)} records)")
    print(f"{'='*60}")

    counts = Counter(r.get("error_type") for r in records)
    total = len(records)

    print("\n--- Error type distribution ---")
    for et in sorted(VALID_TYPES) + ["<missing>", "<invalid>"]:
        if et == "<missing>":
            n = sum(1 for r in records if "error_type" not in r)
        elif et == "<invalid>":
            n = sum(1 for r in records if r.get("error_type") not in VALID_TYPES)
        else:
            n = counts.get(et, 0)
        bar = "#" * int(n / total * 40)
        print(f"  {et:<15} {n:>5}  ({n/total*100:5.1f}%)  {bar}")

    invalid = [r for r in records if r.get("error_type") not in VALID_TYPES]
    if invalid:
        print(f"\n[WARN] {len(invalid)} records with invalid error_type:")
        for r in invalid[:5]:
            print(f"  uuid={r.get('uuid')}  error_type={r.get('error_type')!r}")

    empty_span = [r for r in records if r.get("error_type") != "none" and not r.get("error_span")]
    if empty_span:
        print(f"\n[WARN] {len(empty_span)} non-none records with empty error_span")
    else:
        print("\n[OK] All non-none records have error_span")

    return counts


def print_samples(all_records, n=2):
    rng = random.Random(RANDOM_SEED)
    by_type = {}
    for r in all_records:
        et = r.get("error_type", "<invalid>")
        by_type.setdefault(et, []).append(r)

    print(f"\n{'='*60}")
    print(f"  SAMPLES (n={n} per type, combined corpus)")
    print(f"{'='*60}")

    for et in sorted(VALID_TYPES):
        pool = by_type.get(et, [])
        sample = rng.sample(pool, min(n, len(pool)))
        print(f"\n--- {et.upper()} ({len(pool)} total) ---")
        for r in sample:
            print(f"  SV : {r.get('source_sv', '')[:120]}")
            print(f"  MT : {r.get('mt_en', '')[:120]}")
            print(f"  REF: {r.get('gold_en', '')[:120]}")
            print(f"  ERR: {r.get('error_span', '')[:80]}")
            print(f"  WHY: {r.get('explanation', '')[:120]}")
            print()


def main():
    all_records = []
    totals = Counter()

    for name, path in FILES.items():
        if not path.exists():
            print(f"[ERROR] File not found: {path}")
            continue
        records = load_jsonl(path)
        counts = check_file(name, records)
        totals += counts
        all_records.extend(records)

    if len(FILES) > 1 and all_records:
        total = len(all_records)
        print(f"\n{'='*60}")
        print(f"  COMBINED TOTAL  ({total} records)")
        print(f"{'='*60}")
        for et in sorted(VALID_TYPES):
            n = totals.get(et, 0)
            print(f"  {et:<15} {n:>5}  ({n/total*100:5.1f}%)")

    print_samples(all_records)


if __name__ == "__main__":
    main()
