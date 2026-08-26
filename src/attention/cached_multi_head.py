from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.attention.cached_attention import (
    CachedScaledDotProductAttention,
)
from src.attention.kv_cache import (
    KVCache,
)
from src.attention.qkv_projection import (
    TrainableQKVProjection,
)


@dataclass(frozen=True)
class CachedMultiHeadAttentionResult:
    query: np.ndarray
    key: np.ndarray
    value: np.ndarray
    head_outputs: tuple[np.ndarray, ...]
    concatenated: np.ndarray
    output: np.ndarray


class CachedMultiHeadAttention:
    def __init__(
        self,
        model_dimension: int,
        number_of_heads: int,
        seed: int = 42,
    ) -> None:
        if model_dimension <= 0:
            raise ValueError(
                "model_dimension must be positive."
            )

        if number_of_heads <= 0:
            raise ValueError(
                "number_of_heads must be positive."
            )

        if (
            model_dimension
            % number_of_heads
            != 0
        ):
            raise ValueError(
                "model_dimension must be divisible "
                "by number_of_heads."
            )

        self._model_dimension = model_dimension
        self._number_of_heads = number_of_heads
        self._head_dimension = (
            model_dimension
            // number_of_heads
        )

        self._projection = (
            TrainableQKVProjection(
                input_dimension=model_dimension,
                attention_dimension=model_dimension,
                seed=seed,
            )
        )

        rng = np.random.default_rng(
            seed + 1
        )

        scale = 1.0 / np.sqrt(
            model_dimension
        )

        self._output_weights = rng.normal(
            0.0,
            scale,
            size=(
                model_dimension,
                model_dimension,
            ),
        )

        self._caches = tuple(
            KVCache(
                key_dimension=self._head_dimension,
                value_dimension=self._head_dimension,
            )
            for _ in range(
                number_of_heads
            )
        )

        self._cached_attention = tuple(
            CachedScaledDotProductAttention(
                key_dimension=self._head_dimension
            )
            for _ in range(
                number_of_heads
            )
        )

    @property
    def model_dimension(self) -> int:
        return self._model_dimension

    @property
    def number_of_heads(self) -> int:
        return self._number_of_heads

    @property
    def head_dimension(self) -> int:
        return self._head_dimension

    @property
    def cache_length(self) -> int:
        if not self._caches:
            return 0

        return self._caches[0].sequence_length

    @property
    def projection(
        self,
    ) -> TrainableQKVProjection:
        return self._projection

    @property
    def output_weights(self) -> np.ndarray:
        return self._output_weights.copy()

    def reset_cache(self) -> None:
        for cache in self._caches:
            cache.clear()

    def truncate_cache_left(
        self,
        count: int,
    ) -> None:
        if count < 0:
            raise ValueError(
                "Truncation count cannot be negative."
            )

        if count > self.cache_length:
            raise ValueError(
                "Truncation count cannot exceed cache length."
            )

        for cache in self._caches:
            cache.truncate_left(
                count
            )

    def forward_token(
        self,
        inputs: np.ndarray,
    ) -> CachedMultiHeadAttentionResult:
        values = np.asarray(
            inputs,
            dtype=np.float64,
        )

        if values.ndim != 2:
            raise ValueError(
                "Inputs must be a two-dimensional matrix."
            )

        if values.shape != (
            1,
            self._model_dimension,
        ):
            raise ValueError(
                "Cached attention expects exactly one "
                "token with model_dimension features."
            )

        projection = self._projection.forward(
            values
        )

        queries = self._split_heads(
            projection.queries
        )

        keys = self._split_heads(
            projection.keys
        )

        value_heads = self._split_heads(
            projection.values
        )

        head_outputs: list[np.ndarray] = []

        for index in range(
            self._number_of_heads
        ):
            self._caches[index].append(
                keys[index],
                value_heads[index],
            )

            output = (
                self._cached_attention[index].attend(
                    queries[index],
                    self._caches[index],
                )
            )

            head_outputs.append(
                output
            )

        concatenated = np.concatenate(
            head_outputs,
            axis=1,
        )

        output = (
            concatenated
            @ self._output_weights
        )

        return CachedMultiHeadAttentionResult(
            query=projection.queries,
            key=projection.keys,
            value=projection.values,
            head_outputs=tuple(
                head_outputs
            ),
            concatenated=concatenated,
            output=output,
        )

    def _split_heads(
        self,
        values: np.ndarray,
    ) -> tuple[np.ndarray, ...]:
        reshaped = values.reshape(
            1,
            self._number_of_heads,
            self._head_dimension,
        )

        return tuple(
            reshaped[
                :,
                index,
                :,
            ]
            for index in range(
                self._number_of_heads
            )
        )
