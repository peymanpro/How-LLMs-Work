from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SamplingResult:
    token_id: int
    probabilities: np.ndarray


class TemperatureSampler:
    def __init__(
        self,
        seed: int | None = None,
    ) -> None:
        self._rng = np.random.default_rng(seed)

    def sample(
        self,
        logits: list[float] | np.ndarray,
        temperature: float,
    ) -> SamplingResult:
        if temperature <= 0.0:
            raise ValueError(
                "Temperature must be positive."
            )

        values = np.asarray(
            logits,
            dtype=np.float64,
        )

        if values.ndim != 1:
            raise ValueError(
                "Logits must be a one-dimensional vector."
            )

        if values.size == 0:
            raise ValueError(
                "Logits cannot be empty."
            )

        scaled = values / temperature

        shifted = (
            scaled
            - np.max(scaled)
        )

        probabilities = np.exp(
            shifted
        )

        probabilities /= np.sum(
            probabilities
        )

        token_id = int(
            self._rng.choice(
                len(probabilities),
                p=probabilities,
            )
        )

        return SamplingResult(
            token_id=token_id,
            probabilities=probabilities,
        )
