from __future__ import annotations

import random
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from data.loader import Segment, Term


class ErrorType(str, Enum):
    TERMINOLOGY = "terminology"
    NUMERACY = "numeracy"
    NAMED_ENTITY = "named_entity"
    FLUENCY = "fluency"
    STYLE_GUIDE = "style_guide"
    CONSISTENCY = "consistency"


@dataclass
class EvalSegment:
    """A segment with a single controlled error injected into the draft."""
    segment_id: str
    source: str
    draft: str            # injected (erroneous) draft
    gold_translation: str # clean human reference
    error_type: ErrorType
    injected_span: str    # the exact part of the draft that contains the error
    error_description: str
    is_novel: bool = True
    is_clean: bool = False  # True for uninjected segments used to measure FP/TN


# Client style-guide rules (from Client Financial Style Guide, March 2026)
# Format: (wrong, correct) — injector replaces `correct` with `wrong` in the draft
_STYLE_SWAPS = [
    ("per cent", "percent"),           # docx: "Use percent (not per cent)"
    ("square meters", "m²"),           # docx: use m²/m2, not square meters
    ("square meters", "m2"),
    ("December 31", "31 December"),    # docx: British day-first date format
    ("nomination committee", "Nomination Committee"),   # docx: capitalise
    ("remuneration committee", "Remuneration Committee"),
    ("audit committee", "Audit Committee"),
    ("significant accounting policies", "material accounting policy information"),  # IAS 1 amendment
]

# IFRS near-miss term pairs: (wrong, correct) — from financial domain knowledge
_IFRS_NEAR_MISSES: list[tuple[str, str]] = [
    ("depreciation", "impairment"),
    ("impairment", "depreciation"),
    ("income", "revenue"),
    ("revenue", "income"),
    ("debt", "liability"),
    ("liability", "debt"),
    ("cost", "expense"),
    ("expense", "cost"),
    ("profit", "earnings"),
    ("earnings", "profit"),
    ("provision", "accrual"),
    ("accrual", "provision"),
    ("amortisation", "depreciation"),
    ("depreciation", "amortisation"),
    ("equity", "capital"),
    ("capital", "equity"),
    ("write-down", "impairment loss"),
    ("impairment loss", "write-down"),
]

_NE_SWAPS: list[tuple[str, str]] = [
    # Company names
    ("WDP", "Client"),
    ("Elgiganten", "Client"),
    # Swedish logistics cities
    ("Jönköping", "Norrköping"),
    ("Norrköping", "Jönköping"),
    ("Gothenburg", "Stockholm"),
    ("Stockholm", "Gothenburg"),
    ("Malmö", "Gothenburg"),
    ("Gothenburg", "Malmö"),
    # Market segment
    ("Mid Cap", "Large Cap"),
    ("Large Cap", "Mid Cap"),
    # Stock exchange
    ("Nasdaq Copenhagen", "Nasdaq Stockholm"),
    # Certification standard
    ("ISO 9001", "ISO 14001"),
]

_NUMBER_PATTERN = re.compile(r"\b(\d[\d,\.]*)\s*(million|billion|thousand|MSEK|KSEK|SEK|EUR|USD|%)\b", re.I)
_DATE_PATTERN = re.compile(r"\b(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})\b")


def inject_terminology(
    segment: Segment,
    termbase: list[Term],
    rng: random.Random,
) -> Optional[EvalSegment]:
    """
    Swaps a correct IFRS term in the translation for a plausible near-miss.
    Uses the near-miss table first; falls back to termbase swaps.
    """
    draft = segment.translation

    # Try IFRS near-miss swaps
    rng.shuffle(_IFRS_NEAR_MISSES)
    for wrong, correct in _IFRS_NEAR_MISSES:
        pattern = re.compile(r"\b" + re.escape(correct) + r"\b", re.I)
        if pattern.search(draft):
            injected = pattern.sub(wrong, draft, count=1)
            if injected != draft:
                return EvalSegment(
                    segment_id=segment.id,
                    source=segment.source,
                    draft=injected,
                    gold_translation=segment.translation,
                    error_type=ErrorType.TERMINOLOGY,
                    injected_span=wrong,
                    error_description=f"'{correct}' replaced with '{wrong}'",
                )

    # Fallback: swap a termbase English term for its Swedish source
    rng.shuffle(termbase)
    for term in termbase:
        for en_form in term.all_en():
            pattern = re.compile(r"\b" + re.escape(en_form) + r"\b", re.I)
            if pattern.search(draft) and term.sv:
                injected = pattern.sub(term.sv, draft, count=1)
                if injected != draft:
                    return EvalSegment(
                        segment_id=segment.id,
                        source=segment.source,
                        draft=injected,
                        gold_translation=segment.translation,
                        error_type=ErrorType.TERMINOLOGY,
                        injected_span=term.sv,
                        error_description=f"'{en_form}' replaced with SV term '{term.sv}'",
                    )
    return None


def inject_numeracy(segment: Segment, rng: random.Random) -> Optional[EvalSegment]:
    """Corrupts a number, magnitude, or date in the translation."""
    draft = segment.translation

    # Try magnitude corruption: "5.2 million" → "5.2 thousand"
    m = _NUMBER_PATTERN.search(draft)
    if m:
        orig_magnitude = m.group(2)
        magnitudes = ["million", "billion", "thousand"]
        alternatives = [x for x in magnitudes if x.lower() != orig_magnitude.lower()]
        if alternatives:
            new_magnitude = rng.choice(alternatives)
            injected = draft[:m.start(2)] + new_magnitude + draft[m.end(2):]
            return EvalSegment(
                segment_id=segment.id,
                source=segment.source,
                draft=injected,
                gold_translation=segment.translation,
                error_type=ErrorType.NUMERACY,
                injected_span=new_magnitude,
                error_description=f"magnitude '{orig_magnitude}' replaced with '{new_magnitude}'",
            )

    # Try date corruption: shift day by ±1
    m = _DATE_PATTERN.search(draft)
    if m:
        day = int(m.group(1))
        wrong_day = day - 1 if day > 1 else day + 1
        injected = draft[:m.start(1)] + str(wrong_day) + draft[m.end(1):]
        return EvalSegment(
            segment_id=segment.id,
            source=segment.source,
            draft=injected,
            gold_translation=segment.translation,
            error_type=ErrorType.NUMERACY,
            injected_span=str(wrong_day),
            error_description=f"date day {day} shifted to {wrong_day}",
        )

    return None


def inject_fluency(segment: Segment, rng: random.Random) -> Optional[EvalSegment]:
    """
    Introduces a grammatical / fluency error by removing 'was', 'were',
    or swapping article 'a'/'an', or removing 'the'.
    """
    draft = segment.translation
    candidates = []

    # Drop auxiliary "was/were"
    for aux in ["was recognised", "were recognised", "was recorded", "is recognised"]:
        if aux in draft:
            candidates.append((aux, aux.split(" ", 1)[1]))

    # Drop "the" before a noun phrase
    for m in re.finditer(r"\bthe ([A-Z][a-z]+)", draft):
        candidates.append((m.group(0), m.group(1)))

    if candidates:
        orig, replacement = rng.choice(candidates)
        injected = draft.replace(orig, replacement, 1)
        return EvalSegment(
            segment_id=segment.id,
            source=segment.source,
            draft=injected,
            gold_translation=segment.translation,
            error_type=ErrorType.FLUENCY,
            injected_span=replacement,
            error_description=f"fluency: '{orig}' → '{replacement}'",
        )
    return None


def inject_style_guide(segment: Segment, rng: random.Random) -> Optional[EvalSegment]:
    """Violates a Client style-guide rule."""
    draft = segment.translation
    rng_swaps = list(_STYLE_SWAPS)
    rng.shuffle(rng_swaps)
    for wrong, correct in rng_swaps:
        if correct in draft:
            injected = draft.replace(correct, wrong, 1)
            if injected != draft:
                return EvalSegment(
                    segment_id=segment.id,
                    source=segment.source,
                    draft=injected,
                    gold_translation=segment.translation,
                    error_type=ErrorType.STYLE_GUIDE,
                    injected_span=wrong,
                    error_description=f"style: '{correct}' → '{wrong}'",
                )
    return None


def inject_consistency(segment: Segment, rng: random.Random) -> Optional[EvalSegment]:
    """
    Introduces a consistency error by negating a key financial claim:
    swaps 'increased' ↔ 'decreased', 'profit' ↔ 'loss', etc.
    """
    draft = segment.translation
    consistency_swaps = [
        ("increased", "decreased"), ("decreased", "increased"),
        ("profit", "loss"), ("loss", "profit"),
        ("positive", "negative"), ("negative", "positive"),
        ("exceeded", "fell below"), ("fell below", "exceeded"),
        ("above", "below"), ("below", "above"),
    ]
    rng.shuffle(consistency_swaps)
    for wrong, correct in consistency_swaps:
        if re.search(r"\b" + re.escape(correct) + r"\b", draft, re.I):
            injected = re.sub(r"\b" + re.escape(correct) + r"\b", wrong, draft, count=1, flags=re.I)
            if injected != draft:
                return EvalSegment(
                    segment_id=segment.id,
                    source=segment.source,
                    draft=injected,
                    gold_translation=segment.translation,
                    error_type=ErrorType.CONSISTENCY,
                    injected_span=wrong,
                    error_description=f"consistency: '{correct}' → '{wrong}'",
                )
    return None


def inject_named_entity(segment: Segment, rng: random.Random) -> Optional[EvalSegment]:
    draft = segment.translation
    candidates = list(_NE_SWAPS)
    rng.shuffle(candidates)
    for wrong, correct in candidates:
        pattern = re.compile(r"\b" + re.escape(correct) + r"\b", re.I)
        if pattern.search(draft):
            injected = pattern.sub(wrong, draft, count=1)
            if injected != draft:
                return EvalSegment(
                    segment_id=segment.id,
                    source=segment.source,
                    draft=injected,
                    gold_translation=segment.translation,
                    error_type=ErrorType.NAMED_ENTITY,
                    injected_span=wrong,
                    error_description=f"named entity: '{correct}' replaced with '{wrong}'",
                )
    return None


def build_eval_set(
    segments: list[Segment],
    termbase: list[Term],
    n_per_type: int = 200,
    seed: int = 42,
) -> tuple[list[EvalSegment], set[str]]:
    """
    Builds a balanced evaluation set: n_per_type injected segments per error type.
    Segments are drawn from the novel (non-TM-matched) pool.
    Returns (injected_set, used_ids) so the caller can build a disjoint clean set.
    """
    rng = random.Random(seed)
    shuffled = list(segments)
    rng.shuffle(shuffled)

    injectors = {
        ErrorType.TERMINOLOGY: lambda s: inject_terminology(s, termbase, rng),
        ErrorType.NUMERACY: lambda s: inject_numeracy(s, rng),
        ErrorType.NAMED_ENTITY: lambda s: inject_named_entity(s, rng),
        ErrorType.FLUENCY: lambda s: inject_fluency(s, rng),
        ErrorType.STYLE_GUIDE: lambda s: inject_style_guide(s, rng),
        ErrorType.CONSISTENCY: lambda s: inject_consistency(s, rng),
    }

    eval_set: list[EvalSegment] = []
    used_ids: set[str] = set()
    for error_type, inject_fn in injectors.items():
        count = 0
        for seg in shuffled:
            if count >= n_per_type:
                break
            if seg.id in used_ids:
                continue
            result = inject_fn(seg)
            if result is not None:
                eval_set.append(result)
                used_ids.add(seg.id)
                count += 1
        print(f"  {error_type.value}: {count}/{n_per_type} segments injected")

    return eval_set, used_ids


def build_clean_set(
    segments: list[Segment],
    n_clean: int = 200,
    seed: int = 42,
    exclude_ids: set[str] | None = None,
) -> list[EvalSegment]:
    """
    Returns n_clean uninjected segments for false-positive / true-negative measurement.
    Segments are drawn from the pool excluding any ids already used for injection.
    One clean segment produces TN or FP signals for all 6 error types simultaneously.
    """
    rng = random.Random(seed + 1)  # offset seed so order differs from injected set
    candidates = [s for s in segments if exclude_ids is None or s.id not in exclude_ids]
    rng.shuffle(candidates)
    clean_set = []
    for seg in candidates[:n_clean]:
        clean_set.append(EvalSegment(
            segment_id=f"{seg.id}_clean",
            source=seg.source,
            draft=seg.translation,
            gold_translation=seg.translation,
            error_type=ErrorType.TERMINOLOGY,  # placeholder — ignored; is_clean=True governs
            injected_span="",
            error_description="clean segment (no injection)",
            is_novel=True,
            is_clean=True,
        ))
    print(f"  clean: {len(clean_set)}/{n_clean} segments sampled")
    return clean_set
