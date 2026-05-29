from __future__ import annotations

from dataclasses import dataclass

# xCOMET integration — requires: pip install unbabel-comet
# Model: Unbabel/xCOMET-XL (GPU) or Unbabel/xCOMET-Lite (CPU fallback)


@dataclass
class XCOMETScore:
    segment_score: float       # segment-level score [0, 1]
    system_score: float        # corpus-level average
    error_spans: list[dict]    # span-level annotations from xCOMET


class XCOMETScorer:
    """
    Wraps the unbabel-comet xCOMET model for span-level translation scoring.

    Usage:
        scorer = XCOMETScorer(model_id="Unbabel/xCOMET-XL")
        scores = scorer.score_batch(sources, hypotheses, references)

    Implementation notes:
        - Load model once, reuse across all grid cells.
        - Batch size 8 is a good default for 16GB VRAM.
        - For CPU: use xCOMET-Lite, batch_size=1.
        - xCOMET outputs both segment scores and word-level error spans with severity.

    August implementation task:
        pip install unbabel-comet
        from comet import download_model, load_from_checkpoint
        model_path = download_model("Unbabel/xCOMET-XL")
        model = load_from_checkpoint(model_path)
        data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(sources, hypotheses, refs)]
        output = model.predict(data, batch_size=8, gpus=1)
    """

    def __init__(self, model_id: str = "Unbabel/xCOMET-XL", device: str = "auto"):
        self.model_id = model_id
        self.device = device
        self._model = None

    def _load(self) -> None:
        raise NotImplementedError(
            "xCOMET scorer not yet loaded. "
            "Install unbabel-comet and implement August task."
        )

    def score_batch(
        self,
        sources: list[str],
        hypotheses: list[str],
        references: list[str],
        batch_size: int = 8,
    ) -> list[XCOMETScore]:
        raise NotImplementedError("xCOMET scorer — August task.")

    def score_single(self, source: str, hypothesis: str, reference: str) -> XCOMETScore:
        results = self.score_batch([source], [hypothesis], [reference], batch_size=1)
        return results[0]
