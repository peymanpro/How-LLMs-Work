from __future__ import annotations

import numpy as np

from src.attention.kv_cache import (
    KVCache,
)


class CachedScaledDotProductAttention:
    def __init__(
        self,
        key_dimension: int,
    ) -> None:
        if key_dimension <= 0:
            raise ValueError(
                "key_dimension must be positive."
            )

        self._key_dimension = key_dimension

    def attend(
        self,
        query: np.ndarray,
        cache: KVCache,
    ) -> np.ndarray:
        q = np.asarray(
            query,
            dtype=np.float64,
        )

        if q.ndim != 2:
            raise ValueError(
                "Query must be a two-dimensional matrix."
            )

        if q.shape[1] != self._key_dimension:
            raise ValueError(
                "Query dimension does not match key dimension."
            )

        if q.shape[0] != 1:
            raise ValueError(
                "Cached attention currently expects one query "
                "position at a time."
            )

        state = cache.get()

        if state.keys.shape[0] == 0:
            raise ValueError(
                "KV cache cannot be empty."
            )

        scores = (
            q
            @ state.keys.T
            / np.sqrt(
                self._key_dimension
            )
        )

        probabilities = self._softmax(
            scores
        )

        return (
            probabilities
            @ state.values
        )

    @staticmethod
    def _softmax(
        values: np.ndarray,
    ) -> np.ndarray:
        maximum = np.max(
            values,
            axis=1,
            keepdims=True,
        )

        exponentials = np.exp(
            values
            - maximum
        )

        return (
            exponentials
            / np.sum(
                exponentials,
                axis=1,
                keepdims=True,
            )
        )
