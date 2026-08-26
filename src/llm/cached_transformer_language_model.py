from __future__ import annotations

import numpy as np

from src.llm.cached_transformer_backbone import (
    CachedTransformerBackbone,
)


class CachedTransformerLanguageModel:
    def __init__(
        self,
        backbone: CachedTransformerBackbone,
        vocabulary_size: int,
        seed: int = 42,
    ) -> None:
        if vocabulary_size <= 0:
            raise ValueError(
                "Vocabulary size must be positive."
            )

        self._backbone = backbone
        self._vocabulary_size = vocabulary_size

        rng = np.random.default_rng(
            seed
        )

        scale = 1.0 / np.sqrt(
            backbone.model_dimension
        )

        self._output_weights = rng.normal(
            0.0,
            scale,
            size=(
                backbone.model_dimension,
                vocabulary_size,
            ),
        )

        self._output_bias = np.zeros(
            vocabulary_size,
            dtype=np.float64,
        )

    @property
    def vocabulary_size(self) -> int:
        return self._vocabulary_size

    @property
    def context_size(self) -> int:
        return self._backbone.context_size

    @property
    def model_dimension(self) -> int:
        return self._backbone.model_dimension

    @property
    def cache_length(self) -> int:
        return self._backbone.cache_length

    def reset_cache(self) -> None:
        self._backbone.reset_cache()

    def logits(
        self,
        token_ids: list[int],
    ) -> np.ndarray:
        hidden = self._backbone.forward(
            token_ids
        )

        return self._project(
            hidden
        )

    def incremental_logits(
        self,
        token_id: int,
    ) -> np.ndarray:
        hidden = self._backbone.incremental_forward(
            token_id
        )

        return self._project(
            hidden
        )[0]

    def _project(
        self,
        hidden: np.ndarray,
    ) -> np.ndarray:
        expected_shape = (
            hidden.shape[0],
            self.model_dimension,
        )

        if hidden.shape != expected_shape:
            raise ValueError(
                "Backbone returned an unexpected hidden-state shape."
            )

        return (
            hidden @ self._output_weights
            + self._output_bias
        )
