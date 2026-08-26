from __future__ import annotations

import numpy as np

from src.attention.cached_multi_head import (
    CachedMultiHeadAttention,
)


class CachedTransformerBackbone:
    def __init__(
        self,
        vocabulary_size: int,
        model_dimension: int,
        number_of_heads: int,
        context_size: int,
        seed: int = 42,
    ) -> None:
        if vocabulary_size <= 0:
            raise ValueError(
                "Vocabulary size must be positive."
            )

        if model_dimension <= 0:
            raise ValueError(
                "Model dimension must be positive."
            )

        if number_of_heads <= 0:
            raise ValueError(
                "Number of heads must be positive."
            )

        if context_size <= 0:
            raise ValueError(
                "Context size must be positive."
            )

        if model_dimension % number_of_heads != 0:
            raise ValueError(
                "Model dimension must be divisible by number of heads."
            )

        self._vocabulary_size = vocabulary_size
        self._model_dimension = model_dimension
        self._context_size = context_size

        rng = np.random.default_rng(seed)

        scale = 1.0 / np.sqrt(
            model_dimension
        )

        self._embeddings = rng.normal(
            0.0,
            scale,
            size=(
                vocabulary_size,
                model_dimension,
            ),
        )

        self._attention = (
            CachedMultiHeadAttention(
                model_dimension=model_dimension,
                number_of_heads=number_of_heads,
                seed=seed,
            )
        )

        self._position = 0

    @property
    def model_dimension(self) -> int:
        return self._model_dimension

    @property
    def context_size(self) -> int:
        return self._context_size

    @property
    def position(self) -> int:
        return self._position

    @property
    def cache_length(self) -> int:
        return self._attention.cache_length

    def reset_cache(self) -> None:
        self._attention.reset_cache()
        self._position = 0

    def incremental_forward(
        self,
        token_id: int,
    ) -> np.ndarray:
        if not 0 <= token_id < self._vocabulary_size:
            raise ValueError(
                "Token ID is outside the vocabulary."
            )

        if self._position >= self._context_size:
            raise ValueError(
                "Context size has been exceeded."
            )

        embedding = self._embeddings[
            token_id
        ].reshape(
            1,
            self._model_dimension,
        )

        position = self._position

        # Simple deterministic positional signal.
        positional = np.zeros_like(
            embedding
        )

        positional[0, position % self._model_dimension] = 1.0

        hidden = (
            embedding
            + positional
        )

        result = self._attention.forward_token(
            hidden
        )

        self._position += 1

        return np.asarray(
            result.output,
            dtype=np.float64,
        ).copy()

    def forward(
        self,
        token_ids: list[int],
    ) -> np.ndarray:
        if not token_ids:
            raise ValueError(
                "Token IDs cannot be empty."
            )

        if len(token_ids) > self._context_size:
            raise ValueError(
                "Input sequence exceeds context size."
            )

        self.reset_cache()

        outputs: list[np.ndarray] = []

        for token_id in token_ids:
            outputs.append(
                self.incremental_forward(
                    token_id
                )
            )

        return np.concatenate(
            outputs,
            axis=0,
        )
