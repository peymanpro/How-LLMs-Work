from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.attention.qkv_projection import (
    QKVProjectionResult,
    TrainableQKVProjection,
)
from src.attention.scaled_dot_product import (
    AttentionResult,
    ScaledDotProductAttention,
)


@dataclass(frozen=True)
class MultiHeadAttentionResult:
    inputs: np.ndarray
    projection: QKVProjectionResult
    heads: tuple[AttentionResult, ...]
    concatenated: np.ndarray
    output: np.ndarray


class TrainableMultiHeadAttention:
    def __init__(
        self,
        model_dimension: int,
        number_of_heads: int,
        seed: int = 42,
        causal: bool = False,
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

        self._causal = causal

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
    def output_weights(self) -> np.ndarray:
        return self._output_weights.copy()

    @property
    def projection(
        self,
    ) -> TrainableQKVProjection:
        return self._projection

    def forward(
        self,
        inputs: np.ndarray,
    ) -> MultiHeadAttentionResult:
        values = np.asarray(
            inputs,
            dtype=np.float64,
        )

        if values.ndim != 2:
            raise ValueError(
                "Inputs must be a two-dimensional matrix."
            )

        if values.shape[1] != self._model_dimension:
            raise ValueError(
                "Input dimension does not match model dimension."
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

        heads: list[AttentionResult] = []

        for query, key, value in zip(
            queries,
            keys,
            value_heads,
            strict=True,
        ):
            heads.append(
                ScaledDotProductAttention(
                    causal=self._causal
                ).forward(
                    query,
                    key,
                    value,
                )
            )

        concatenated = np.concatenate(
            [
                head.output
                for head in heads
            ],
            axis=1,
        )

        output = (
            concatenated
            @ self._output_weights
        )

        return MultiHeadAttentionResult(
            inputs=values,
            projection=projection,
            heads=tuple(heads),
            concatenated=concatenated,
            output=output,
        )

    def _split_heads(
        self,
        values: np.ndarray,
    ) -> tuple[np.ndarray, ...]:
        sequence_length = values.shape[0]

        reshaped = values.reshape(
            sequence_length,
            self._number_of_heads,
            self._head_dimension,
        )

        return tuple(
            reshaped[:, index, :]
            for index in range(
                self._number_of_heads
            )
        )
