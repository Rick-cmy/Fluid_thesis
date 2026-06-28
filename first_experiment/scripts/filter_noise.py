"""
filter_noise.py — Remove alignment-noise records from labelled MT eval files.

Noise criterion: mt_en and gold_en share almost no words AND almost no characters,
meaning the RTF parser picked up text from the wrong table cell.

Heuristics (record is noise if BOTH pass):
  1. word_jaccard(mt_en, gold_en) < WORD_THRESH
  2. char_ratio(mt_en, gold_en)   < CHAR_THRESH   (difflib SequenceMatcher)

Dry-run by default; use --apply to rewrite files in-place.

Usage:
  python scripts/filter_noise.py              # show what would be removed
  python scripts/filter_noise.py --apply      # rewrite labelled files
  python scripts/filter_noise.py --word 0.08 --char 0.3   # tune thresholds
"""

from __future__ import annotations

import argparse
import json
import re
from difflib import SequenceMatcher
from pathlib import Path

THESIS_ROOT = Path(__file__).resolve().parents[2]
MT_DIR = THESIS_ROOT / "data" / "mt_eval"

FILES = [
    MT_DIR / "labelled_q3_2025.jsonl",
    MT_DIR / "labelled_q1_2026.jsonl",
]

DEFAULT_WORD_THRESH = 0.05
DEFAULT_CHAR_THRESH = 0.25

_TAG_RE = re.compile(r"[\[\{]\d+[\]\}]")


def _tokens(text: str) -> set[str]:
    text = _TAG_RE.sub("", text).lower()
    return set(re.findall(r"[a-zåäö0-9]+", text))


def word_jaccard(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def char_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def is_noise(record: dict, word_thresh: float, char_thresh: float) -> bool:
    mt = record.get("mt_en", "")
    gold = record.get("gold_en", "")
    return word_jaccard(mt, gold) < word_thresh and char_ratio(mt, gold) < char_thresh


def process_file(path: Path, word_thresh: float, char_thresh: float, apply: bool) -> None:
    if not path.exists():
        print(f"SKIP (missing): {path}")
        return

    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    noise = [r for r in records if is_noise(r, word_thresh, char_thresh)]
    clean = [r for r in records if not is_noise(r, word_thresh, char_thresh)]

    print(f"\n{path.name}:  {len(records)} total  →  {len(noise)} noise  /  {len(clean)} kept")

    if noise:
        print("  Noise records:")
        for r in noise:
            jac = word_jaccard(r["mt_en"], r["gold_en"])
            cratio = char_ratio(r["mt_en"], r["gold_en"])
            print(f"    [{r.get('error_type','?')}]  jac={jac:.3f}  chr={cratio:.3f}")
            print(f"      SV  : {r.get('source_sv','')[:80]}")
            print(f"      MT  : {r.get('mt_en','')[:80]}")
            print(f"      GOLD: {r.get('gold_en','')[:80]}")

    if apply:
        with path.open("w", encoding="utf-8") as f:
            for r in clean:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  Written {len(clean)} records → {path}")
    else:
        print(f"  (dry run — pass --apply to rewrite)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Rewrite files (default: dry run)")
    parser.add_argument("--word", type=float, default=DEFAULT_WORD_THRESH, help="Word Jaccard threshold")
    parser.add_argument("--char", type=float, default=DEFAULT_CHAR_THRESH, help="Char ratio threshold")
    args = parser.parse_args()

    print(f"Thresholds: word_jaccard < {args.word}  AND  char_ratio < {args.char}")
    print(f"Mode: {'APPLY (rewrite)' if args.apply else 'dry run'}")

    for path in FILES:
        process_file(path, args.word, args.char, args.apply)

    print()


if __name__ == "__main__":
    main()
