#!/usr/bin/env python3
"""
Build style_rules.jsonl for rich_rag from two confirmed sources:
  1. injector.py _STYLE_SWAPS (8 rules, docx-confirmed)
  2. generate_violations.py V01–V30 (style_classifier project, additional coverage)

Output: data/client_baseline/processed/style_rules.jsonl
Each line: {"id", "wrong", "correct", "rule_text", "embed_text", "source"}

embed_text = wrong + correct + rule_text concatenated — what gets indexed in FAISS.
"""
from __future__ import annotations

import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Rule definitions
# ---------------------------------------------------------------------------
# Each tuple: (id, wrong_form, correct_form, rule_text, source_tag)
# wrong_form  = what may appear INCORRECTLY in a draft translation
# correct_form = what the Client style guide requires instead
# ---------------------------------------------------------------------------

_RULES: list[tuple[str, str, str, str, str]] = [
    # ── From injector.py _STYLE_SWAPS (docx-confirmed, March 2026) ──────────
    (
        "INJ-01",
        "per cent",
        "percent",
        "Use 'percent' (not 'per cent') — Client style guide",
        "client_style_guide_docx",
    ),
    (
        "INJ-02",
        "square meters",
        "m²",
        "Use the symbol 'm²' or 'm2' (not 'square meters') for area units",
        "client_style_guide_docx",
    ),
    (
        "INJ-03",
        "December 31",
        "31 December",
        "Use British day-first date format: '31 December' not 'December 31'",
        "client_style_guide_docx",
    ),
    (
        "INJ-04",
        "nomination committee",
        "Nomination Committee",
        "Capitalise 'Nomination Committee' (proper noun)",
        "client_style_guide_docx",
    ),
    (
        "INJ-05",
        "remuneration committee",
        "Remuneration Committee",
        "Capitalise 'Remuneration Committee' (proper noun)",
        "client_style_guide_docx",
    ),
    (
        "INJ-06",
        "audit committee",
        "Audit Committee",
        "Capitalise 'Audit Committee' (proper noun)",
        "client_style_guide_docx",
    ),
    (
        "INJ-07",
        "significant accounting policies",
        "material accounting policy information",
        "Use 'material accounting policy information' (IAS 1 amended 2021), not 'significant accounting policies'",
        "client_style_guide_docx",
    ),
    # ── From generate_violations.py V01–V30 (style_classifier project) ──────
    # V01 covered by INJ-01 above
    (
        "V02",
        "web site",
        "website",
        "Use 'website' as one word (not 'web site')",
        "client_style_guide_classifier",
    ),
    (
        "V03",
        "Mkr",
        "SEK million",
        "Use 'SEK X million' format (not 'X Mkr') in English translations",
        "client_style_guide_classifier",
    ),
    # V04 covered by INJ-02 above
    (
        "V07",
        "occupancy ratio",
        "occupancy rate",
        "Use 'occupancy rate' (not 'occupancy ratio')",
        "client_style_guide_classifier",
    ),
    (
        "V14",
        "weighted average lease expiry",
        "average contract period",
        "Use 'average contract period' (not 'weighted average lease expiry') in Client context",
        "client_style_guide_classifier",
    ),
    (
        "V15",
        "land properties",
        "land holdings",
        "Use 'land holdings' (not 'land properties')",
        "client_style_guide_classifier",
    ),
    (
        "V16",
        "building frames",
        "carcasses",
        "Use 'carcasses' for building shells (not 'building frames') — Client property terminology",
        "client_style_guide_classifier",
    ),
    (
        "V17",
        "rights issue",
        "new share issue",
        "Use 'new share issue' (not 'rights issue') in Client context",
        "client_style_guide_classifier",
    ),
    (
        "V18",
        "property tax value",
        "taxable values",
        "Use 'taxable values' (not 'property tax value')",
        "client_style_guide_classifier",
    ),
    (
        "V19",
        "Profit from investments in",
        "Profit from participations in",
        "Use 'Profit from participations in' (not 'Profit from investments in')",
        "client_style_guide_classifier",
    ),
    (
        "V20",
        "Equity/assets ratio",
        "Equity ratio",
        "Use 'Equity ratio' (not 'Equity/assets ratio') in Client annual reports",
        "client_style_guide_classifier",
    ),
    (
        "V21",
        "EPRA key figures",
        "EPRA performance measures",
        "Use 'EPRA performance measures' (not 'EPRA key figures')",
        "client_style_guide_classifier",
    ),
    (
        "V22",
        "Net profit for the year",
        "Profit for the year",
        "Use 'Profit for the year' (not 'Net profit for the year')",
        "client_style_guide_classifier",
    ),
    (
        "V23",
        "Foreign currency transactions and balances",
        "Foreign currency transactions and balance sheet items",
        "Use 'Foreign currency transactions and balance sheet items' (not '...and balances')",
        "client_style_guide_classifier",
    ),
    (
        "V24",
        "Board fees",
        "Directors' fees",
        "Use 'Directors' fees' (not 'Board fees')",
        "client_style_guide_classifier",
    ),
    (
        "V25",
        "valuation hierarchy",
        "measurement hierarchy",
        "Use 'measurement hierarchy' (IFRS 13 term), not 'valuation hierarchy'",
        "client_style_guide_classifier",
    ),
    (
        "V26",
        "capitalised interest rate",
        "capitalised interest",
        "Use 'capitalised interest' (not 'capitalised interest rate')",
        "client_style_guide_classifier",
    ),
    (
        "V27",
        "cold storage",
        "refrigeration and freezing facility",
        "Use 'refrigeration and freezing facility' (not 'cold storage') — Client property type",
        "client_style_guide_classifier",
    ),
    (
        "V28",
        "live-streamed",
        "presented online",
        "Use 'presented online' (not 'live-streamed') for virtual meetings",
        "client_style_guide_classifier",
    ),
    (
        "V30",
        "key performance indicators",
        "key financial figures",
        "Use 'key financial figures' (not 'key performance indicators') in Client context",
        "client_style_guide_classifier",
    ),
    # V09 covered by INJ-07 above
    (
        "V10",
        "Parent Company financial statements",
        "Parent Company's financial statements",
        "Use possessive form: 'Parent Company's financial statements'",
        "client_style_guide_classifier",
    ),
    # V11-V13 covered by INJ-04/05/06 above
]


def main() -> None:
    out_path = (
        Path(__file__).parent.parent.parent
        / "data"
        / "client_baseline"
        / "processed"
        / "style_rules.jsonl"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for rule_id, wrong, correct, rule_text, source in _RULES:
            embed_text = f"{wrong} {correct} {rule_text}"
            record = {
                "id": rule_id,
                "wrong": wrong,
                "correct": correct,
                "rule_text": rule_text,
                "embed_text": embed_text,
                "source": source,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {written} style rules to {out_path}")
    for rule_id, wrong, correct, rule_text, source in _RULES:
        tag = "[docx]" if "docx" in source else "[cls] "
        print(f"  {tag} {rule_id:<10} {wrong!r:40s} → {correct!r}")


if __name__ == "__main__":
    main()
