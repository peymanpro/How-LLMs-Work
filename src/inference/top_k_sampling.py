from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TopKSamplingResult:
    token_id: int
    probabilities: np.ndarray


class TopKSampler:
    def __init__(
        self,
        seed: int | None = None,
    ) -> None:
        self._rng = np.random.default_rng(seed)

    def sample(
        self,
        logits: list[float] | np.ndarray,
        k: int,
        temperature: float = 1.0,
    ) -> TopKSamplingResult:
        if k <= 0:
            raise ValueError(
                "k must be positive."
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

        if k > values.size:
            raise ValueError(
                "k cannot exceed vocabulary size."
            )

        top_indices = np.argsort(
            values
        )[-k:]

        filtered = np.full_like(
            values,
            -np.inf,
        )

        filtered[
            top_indices
        ] = values[
            top_indices
        ]

        scaled = (
            filtered
            / temperature
        )

        maximum = np.max(
            scaled[
                top_indices
            ]
        )

        exponentials = np.exp(
            scaled
            - maximum
        )

        exponentials[
            ~np.isfinite(
                scaled
            )
        ] = 0.0

        total = np.sum(
            exponentials
        )

        if total <= 0.0:
            raise ValueError(
                "Unable to construct top-k probability distribution."
            )

        probabilities = (
            exponentials
            / total
        )

        token_id = int(
            self._rng.choice(
                len(probabilities),
                p=probabilities,
            )
        )

        return TopKSamplingResult(
            token_id=token_id,
            probabilities=probabilities,
        )
