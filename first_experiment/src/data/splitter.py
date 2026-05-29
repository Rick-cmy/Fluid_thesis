from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable

from data.loader import Segment


@dataclass
class SplitResult:
    novel: list[Segment]
    tm_matched: list[Segment]

    def summary(self) -> str:
        total = len(self.novel) + len(self.tm_matched)
        return (
            f"Total: {total} | Novel: {len(self.novel)} "
            f"({100*len(self.novel)/total:.1f}%) | "
            f"TM-matched: {len(self.tm_matched)} "
            f"({100*len(self.tm_matched)/total:.1f}%)"
        )


@dataclass
class DevTestSplit:
    dev: list[Segment]    # ~20% — for prompt tuning / injector debugging
    test: list[Segment]   # ~80% — final evaluation only, do not touch until grid runs

    def summary(self) -> str:
        total = len(self.dev) + len(self.test)
        return (
            f"Dev: {len(self.dev)} ({100*len(self.dev)/total:.1f}%) | "
            f"Test: {len(self.test)} ({100*len(self.test)/total:.1f}%)"
        )


def dev_test_split(
    novel_segments: list[Segment],
    dev_ratio: float = 0.2,
    seed: int = 42,
) -> DevTestSplit:
    """
    Splits novel segments into dev (prompt tuning) and test (final evaluation).

    IMPORTANT: The test split must not be used for prompt iteration or
    injector debugging — only for the final grid experiment. This prevents
    implicit overfitting of prompts to the evaluation data.

    Args:
        novel_segments: segments from split_novel_vs_tm().novel
        dev_ratio: fraction reserved for development (default 0.2)
        seed: fixed seed for reproducibility across runs
    """
    rng = random.Random(seed)
    shuffled = list(novel_segments)
    rng.shuffle(shuffled)
    n_dev = int(len(shuffled) * dev_ratio)
    return DevTestSplit(dev=shuffled[:n_dev], test=shuffled[n_dev:])


def split_novel_vs_tm(
    master_segments: Iterable[Segment],
    quarterly_segments: Iterable[Segment],
    min_source_len: int = 20,
) -> SplitResult:
    """
    Separates master TMX segments into novel vs TM-matched.

    A segment is TM-matched if its source text appears verbatim (after
    normalisation) in any of the quarterly report segment files. This
    controls the repetition confounder: REIT quarterly reports recur
    heavily across quarters, and TM-recalled segments would inflate
    apparent accuracy.

    Args:
        master_segments: all segments from the master TMX.
        quarterly_segments: segments from Q3/Q1 reports (the test period).
        min_source_len: discard very short segments (boilerplate headers).
    """
    quarterly_sources: set[str] = {
        _normalise(s.source)
        for s in quarterly_segments
        if len(s.source) >= min_source_len
    }

    novel: list[Segment] = []
    tm_matched: list[Segment] = []

    for seg in master_segments:
        if len(seg.source) < min_source_len:
            continue
        if _normalise(seg.source) in quarterly_sources:
            tm_matched.append(seg)
        else:
            novel.append(seg)

    return SplitResult(novel=novel, tm_matched=tm_matched)


def _normalise(text: str) -> str:
    return " ".join(text.lower().split())
