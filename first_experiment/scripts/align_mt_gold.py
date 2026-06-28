"""
align_mt_gold.py — Parse MT RTF exports and align with human gold translations.

Reads:
  data/mt_eval/Client Q3 2025 Svensk.idml_eng-GB.rtf
  data/mt_eval/Client Q1 2026 Svensk.idml_eng-GB.rtf
  data/client_baseline/processed/q3_2025_segments.jsonl
  data/client_baseline/processed/q1_2026_segments.jsonl

Writes:
  data/mt_eval/aligned_q3_2025.jsonl
  data/mt_eval/aligned_q1_2026.jsonl

Each output line:
  {uuid, doc_id, source_sv, mt_en, gold_en, has_numbers, has_tags}
"""

from __future__ import annotations

import json
import re
from pathlib import Path

THESIS_ROOT = Path(__file__).resolve().parents[2]
MT_DIR = THESIS_ROOT / "data" / "mt_eval"
GOLD_DIR = THESIS_ROOT / "data" / "client_baseline" / "processed"

REPORTS = [
    {
        "doc_id": "client_q3_2025",
        "mt_file": "Client Q3 2025 Svensk.idml_eng-GB.rtf",
        "gold_file": "q3_2025_segments.jsonl",
        "out_file": "aligned_q3_2025.jsonl",
    },
    {
        "doc_id": "client_q1_2026",
        "mt_file": "Client Q1 2026 Svensk.idml_eng-GB.rtf",
        "gold_file": "q1_2026_segments.jsonl",
        "out_file": "aligned_q1_2026.jsonl",
    },
]

_CPG_RE = re.compile(r"\\ansicpg(\d+)")
_UUID_RE = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)


def _detect_encoding(raw: str) -> str:
    m = _CPG_RE.search(raw[:3000])
    if not m:
        return "cp1252"
    enc = f"cp{m.group(1)}"
    try:
        b"test".decode(enc)
        return enc
    except LookupError:
        return "cp1252"


def _decode_signed_unicode(n: int) -> str:
    if n < 0:
        n += 65536
    try:
        return chr(n)
    except ValueError:
        return ""


def _rtf_to_text(s: str, encoding: str) -> str:
    out = []
    i = 0
    n = len(s)
    uc_skip = 1

    while i < n:
        ch = s[i]
        if ch == "\\":
            i += 1
            if i >= n:
                break
            nxt = s[i]
            if nxt in "\\{}":
                out.append(nxt)
                i += 1
                continue
            if nxt == "'" and i + 2 < n:
                hx = s[i + 1 : i + 3]
                try:
                    out.append(bytes.fromhex(hx).decode(encoding, errors="replace"))
                except Exception:
                    pass
                i += 3
                continue
            if nxt.isalpha():
                start = i
                while i < n and s[i].isalpha():
                    i += 1
                word = s[start:i]
                sign = 1
                if i < n and s[i] in "+-":
                    if s[i] == "-":
                        sign = -1
                    i += 1
                num_start = i
                while i < n and s[i].isdigit():
                    i += 1
                num = sign * int(s[num_start:i]) if i > num_start else None
                if i < n and s[i] == " ":
                    i += 1
                if word == "uc" and num is not None:
                    uc_skip = max(num, 0)
                elif word == "u" and num is not None:
                    out.append(_decode_signed_unicode(num))
                    for _ in range(uc_skip):
                        if i < n:
                            i += 1
                elif word in {"par", "line"}:
                    out.append("\n")
                elif word == "tab":
                    out.append("\t")
                elif word == "endash":
                    out.append("–")
                elif word == "emdash":
                    out.append("—")
                continue
            if nxt == "~":
                out.append(" ")
            elif nxt in {"-", "_"}:
                out.append("-")
            i += 1
            continue
        if ch in "{}":
            i += 1
            continue
        out.append(ch)
        i += 1

    text = "".join(out).replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


_TAG_RE = re.compile(r"[\[\{]\d+[\]\}]")


def _norm_sv(text: str) -> str:
    """Strip CAT-tool inline tags and normalise whitespace for alignment."""
    return re.sub(r"\s+", " ", _TAG_RE.sub("", text)).strip().lower()


def parse_mt_rtf(path: Path) -> list[dict]:
    """Returns list of {source_sv, mt_en} from the MT RTF file."""
    raw = path.read_bytes().decode("latin-1", errors="ignore")
    encoding = _detect_encoding(raw)
    result = []

    for row in raw.split(r"\row"):
        parts = re.split(r"\\cell(?![a-zA-Z])", row)
        if len(parts) < 4:
            continue
        cells = [_clean(_rtf_to_text(p, encoding)) for p in parts[:-1]]
        cells = [c for c in cells if c]
        if len(cells) < 3:
            continue

        uuid_match = _UUID_RE.search(cells[0])
        if not uuid_match:
            continue

        source_sv = cells[1]
        mt_en = cells[2]
        skip = {"English (United Kingdom)", "Swedish", "[1]", ""}
        if mt_en not in skip and source_sv not in skip:
            result.append({"source_sv": source_sv, "mt_en": mt_en})

    return result


def load_gold(path: Path) -> dict[str, dict]:
    """Returns {normalised_sv: record} from human gold JSONL."""
    gold: dict[str, dict] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            key = _norm_sv(r.get("source_sv", ""))
            if key:
                gold[key] = r
    return gold


def align(mt_rows: list[dict], gold: dict[str, dict], doc_id: str) -> list[dict]:
    records = []
    for row in mt_rows:
        key = _norm_sv(row["source_sv"])
        g = gold.get(key)
        if g is None:
            continue
        gold_en = g.get("reference_en", "")
        mt_en = row["mt_en"]
        source_sv = g.get("source_sv", "")
        if not gold_en or not mt_en or mt_en == gold_en:
            continue
        records.append({
            "uuid": g.get("uuid", ""),
            "doc_id": doc_id,
            "source_sv": source_sv,
            "mt_en": mt_en,
            "gold_en": gold_en,
            "has_numbers": bool(re.search(r"\d", source_sv)),
            "has_tags": bool(re.search(r"[\[\{]\d+[\]\}]", source_sv)),
        })
    return records


def main() -> None:
    MT_DIR.mkdir(parents=True, exist_ok=True)

    for report in REPORTS:
        mt_path = MT_DIR / report["mt_file"]
        gold_path = GOLD_DIR / report["gold_file"]
        out_path = MT_DIR / report["out_file"]

        if not mt_path.exists():
            print(f"SKIP (missing): {mt_path}")
            continue
        if not gold_path.exists():
            print(f"SKIP (missing gold): {gold_path}")
            continue

        print(f"\n{report['doc_id']}")
        mt = parse_mt_rtf(mt_path)
        print(f"  MT segments parsed:  {len(mt)}")

        gold = load_gold(gold_path)
        print(f"  Gold segments loaded: {len(gold)}")

        records = align(mt, gold, report["doc_id"])
        print(f"  Aligned pairs (mt≠gold): {len(records)}")

        with out_path.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  Written → {out_path}")

        if records:
            print("  Preview:")
            r = records[0]
            print(f"    uuid:      {r['uuid']}")
            print(f"    source_sv: {r['source_sv'][:80]}")
            print(f"    mt_en:     {r['mt_en'][:80]}")
            print(f"    gold_en:   {r['gold_en'][:80]}")


if __name__ == "__main__":
    main()
