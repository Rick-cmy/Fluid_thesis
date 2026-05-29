from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class Segment:
    id: str
    source: str
    translation: str
    source_lang: str = "sv"
    target_lang: str = "en-GB"
    has_numbers: bool = False
    has_currency: bool = False
    has_date: bool = False
    source_file: str = ""


@dataclass(frozen=True)
class Term:
    sv: str
    en: str
    sv_variants: tuple[str, ...] = field(default_factory=tuple)
    en_variants: tuple[str, ...] = field(default_factory=tuple)
    source: str = "catena"

    def all_sv(self) -> list[str]:
        return [self.sv] + list(self.sv_variants)

    def all_en(self) -> list[str]:
        return [self.en] + list(self.en_variants)


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    source: str
    draft: str
    source_lang: str
    target_lang: str
    domain: str
    target_locale: str
    expected_issue_dimensions: list[str]
    expected_core_issues: list[str]
    ideal_translation: str
    no_error_case: bool


def load_segments(path: str | Path) -> list[Segment]:
    """Handles both master TMX format (source/translation) and quarterly format (source_sv/reference_en)."""
    segments = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            source = d.get("source") or d.get("source_sv", "")
            translation = d.get("translation") or d.get("reference_en", "")
            seg_id = d.get("id") or d.get("segment_id") or d.get("uuid", "")
            if not source or not translation:
                continue
            segments.append(Segment(
                id=str(seg_id),
                source=source,
                translation=translation,
                source_lang=d.get("source_lang", "sv"),
                target_lang=d.get("target_lang", "en-GB"),
                has_numbers=d.get("has_numbers", False),
                has_currency=d.get("has_currency", False),
                has_date=d.get("has_date", False),
                source_file=d.get("source_file", d.get("doc_id", "")),
            ))
    return segments


def load_termbase(path: str | Path) -> list[Term]:
    terms = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            terms.append(Term(
                sv=d["sv"],
                en=d["en"],
                sv_variants=tuple(d.get("sv_variants", [])),
                en_variants=tuple(d.get("en_variants", [])),
                source=d.get("source", "catena"),
            ))
    return terms


def load_benchmark(path: str | Path) -> list[BenchmarkCase]:
    cases = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            cases.append(BenchmarkCase(
                case_id=d["case_id"],
                source=d["source"],
                draft=d["draft"],
                source_lang=d.get("source_lang", "Swedish"),
                target_lang=d.get("target_lang", "English"),
                domain=d.get("domain", "IFRS financial reporting"),
                target_locale=d.get("target_locale", "en-GB"),
                expected_issue_dimensions=d.get("expected_issue_dimensions", []),
                expected_core_issues=d.get("expected_core_issues", []),
                ideal_translation=d.get("ideal_translation", ""),
                no_error_case=d.get("no_error_case", False),
            ))
    return cases
