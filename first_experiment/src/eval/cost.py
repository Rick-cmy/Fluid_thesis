from __future__ import annotations

from dataclasses import dataclass, field


# Approximate cost per 1M tokens (USD) for reference — update per provider
_COST_PER_1M_PROMPT = 0.0      # Ollama local: $0 (electricity only)
_COST_PER_1M_COMPLETION = 0.0


@dataclass
class CostRecord:
    rag_level: str
    coord_level: str
    trial: int
    n_segments: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s_total: float = 0.0
    latency_s_values: list[float] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def cost_usd(self) -> float:
        return (
            self.prompt_tokens / 1_000_000 * _COST_PER_1M_PROMPT
            + self.completion_tokens / 1_000_000 * _COST_PER_1M_COMPLETION
        )

    @property
    def latency_mean_s(self) -> float:
        return self.latency_s_total / self.n_segments if self.n_segments > 0 else 0.0

    @property
    def latency_p95_s(self) -> float:
        if not self.latency_s_values:
            return 0.0
        sorted_vals = sorted(self.latency_s_values)
        idx = int(0.95 * len(sorted_vals))
        return sorted_vals[min(idx, len(sorted_vals) - 1)]

    def add(self, prompt_tokens: int, completion_tokens: int, latency_s: float) -> None:
        self.n_segments += 1
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.latency_s_total += latency_s
        self.latency_s_values.append(latency_s)

    def to_dict(self) -> dict:
        return {
            "rag_level": self.rag_level,
            "coord_level": self.coord_level,
            "trial": self.trial,
            "n_segments": self.n_segments,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "latency_mean_s": round(self.latency_mean_s, 3),
            "latency_p95_s": round(self.latency_p95_s, 3),
        }
