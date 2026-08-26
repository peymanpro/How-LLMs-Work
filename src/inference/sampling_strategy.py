from __future__ import annotations

from typing import Protocol

import numpy as np


class SamplingStrategy(Protocol):
    def sample(
        self,
        logits: np.ndarray,
    ) -> int:
        ...


class GreedySamplingStrategy:
    def sample(
        self,
        logits: np.ndarray,
    ) -> int:
        values = np.asarray(
            logits,
            dtype=np.float64,
        )

        if values.ndim != 1:
            raise ValueError(
                "Logits must be one-dimensional."
            )

        if values.size == 0:
            raise ValueError(
                "Logits cannot be empty."
            )

        return int(
            np.argmax(values)
        )


class TemperatureSamplingStrategy:
    def __init__(
        self,
        temperature: float,
        seed: int | None = None,
    ) -> None:
        if temperature <= 0.0:
            raise ValueError(
                "Temperature must be positive."
            )

        self._temperature = temperature
        self._rng = np.random.default_rng(
            seed
        )

    def sample(
        self,
        logits: np.ndarray,
    ) -> int:
        values = np.asarray(
            logits,
            dtype=np.float64,
        )

        if values.ndim != 1:
            raise ValueError(
                "Logits must be one-dimensional."
            )

        if values.size == 0:
            raise ValueError(
                "Logits cannot be empty."
            )

        scaled = (
            values
            / self._temperature
        )

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

        return int(
            self._rng.choice(
                values.size,
                p=probabilities,
            )
        )


class TopKSamplingStrategy:
    def __init__(
        self,
        k: int,
        temperature: float = 1.0,
        seed: int | None = None,
    ) -> None:
        if k <= 0:
            raise ValueError(
                "k must be positive."
            )

        if temperature <= 0.0:
            raise ValueError(
                "Temperature must be positive."
            )

        self._k = k
        self._temperature = temperature
        self._rng = np.random.default_rng(
            seed
        )

    def sample(
        self,
        logits: np.ndarray,
    ) -> int:
        values = np.asarray(
            logits,
            dtype=np.float64,
        )

        if values.ndim != 1:
            raise ValueError(
                "Logits must be one-dimensional."
            )

        if values.size == 0:
            raise ValueError(
                "Logits cannot be empty."
            )

        if self._k > values.size:
            raise ValueError(
                "k cannot exceed vocabulary size."
            )

        top_indices = np.argsort(
            values
        )[-self._k:]

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
            / self._temperature
        )

        maximum = np.max(
            scaled[
                top_indices
            ]
        )

        probabilities = np.exp(
            scaled
            - maximum
        )

        probabilities[
            ~np.isfinite(
                scaled
            )
        ] = 0.0

        probabilities /= np.sum(
            probabilities
        )

        return int(
            self._rng.choice(
                values.size,
                p=probabilities,
            )
        )


class TopPSamplingStrategy:
    def __init__(
        self,
        p: float,
        temperature: float = 1.0,
        seed: int | None = None,
    ) -> None:
        if not 0.0 < p <= 1.0:
            raise ValueError(
                "p must be in the interval (0, 1]."
            )

        if temperature <= 0.0:
            raise ValueError(
                "Temperature must be positive."
            )

        self._p = p
        self._temperature = temperature
        self._rng = np.random.default_rng(
            seed
        )

    def sample(
        self,
        logits: np.ndarray,
    ) -> int:
        values = np.asarray(
            logits,
            dtype=np.float64,
        )

        if values.ndim != 1:
            raise ValueError(
                "Logits must be one-dimensional."
            )

        if values.size == 0:
            raise ValueError(
                "Logits cannot be empty."
            )

        scaled = (
            values
            / self._temperature
        )

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
            probabilities[
                sorted_indices
            ]
        )

        cumulative = np.cumsum(
            sorted_probabilities
        )

        cutoff = int(
            np.searchsorted(
                cumulative,
                self._p,
                side="left",
            )
        )

        candidate_indices = (
            sorted_indices[
                :cutoff + 1
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

        return int(
            self._rng.choice(
                values.size,
                p=filtered,
            )
        )
