from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TopPSamplingResult:
    token_id: int
    probabilities: np.ndarray


class TopPSampler:
    def __init__(
        self,
        seed: int | None = None,
    ) -> None:
        self._rng = np.random.default_rng(seed)

    def sample(
        self,
        logits: list[float] | np.ndarray,
        p: float,
        temperature: float = 1.0,
    ) -> TopPSamplingResult:
        if not 0.0 < p <= 1.0:
            raise ValueError(
                "p must be in the interval (0, 1]."
            )

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

        sorted_indices = np.argsort(
            probabilities
        )[::-1]

        sorted_probabilities = (
            probabilities[sorted_indices]
        )

        cumulative = np.cumsum(
            sorted_probabilities
        )

        cutoff = int(
            np.searchsorted(
                cumulative,
                p,
                side="left",
            )
        )

        candidate_count = cutoff + 1

        candidate_indices = (
            sorted_indices[
                :candidate_count
            ]
        )

        filtered = np.zeros_like(
            probabilities
        )

        filtered[
            candidate_indices
        ] = probabilities[
            candidate_indices
        ]

        filtered /= np.sum(
            filtered
        )

        token_id = int(
            self._rng.choice(
                len(filtered),
                p=filtered,
            )
        )

        return TopPSamplingResult(
            token_id=token_id,
            probabilities=filtered,
        )
