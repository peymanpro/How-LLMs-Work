from __future__ import annotations

from typing import Any

import numpy as np

from src.llm.transformer_backbone import (
    TransformerBackbone,
)


class HowTransformersWorkBackboneAdapter:
    def __init__(
        self,
        transformer: Any,
        context_size: int,
        model_dimension: int,
    ) -> None:
        if context_size <= 0:
            raise ValueError(
                "Context size must be positive."
            )

        if model_dimension <= 0:
            raise ValueError(
                "Model dimension must be positive."
            )

        self._transformer = transformer
        self._context_size = context_size
        self._model_dimension = model_dimension

    @property
    def model_dimension(self) -> int:
        return self._model_dimension

    @property
    def context_size(self) -> int:
        return self._context_size

    def forward(
        self,
        token_ids: list[int],
    ) -> np.ndarray:
        if len(token_ids) != self._context_size:
            raise ValueError(
                "Input sequence length does not "
                "match context size."
            )

        result = self._transformer.forward(
            token_ids
        )

        hidden = result.decoder_output.data

        expected_shape = (
            self._context_size,
            self._model_dimension,
        )

        if hidden.shape != expected_shape:
            raise ValueError(
                "Transformer decoder output has "
                "an unexpected shape."
            )

        return np.asarray(
            hidden,
            dtype=np.float64,
        ).copy()


def assert_transformer_backbone_contract(
    backbone: HowTransformersWorkBackboneAdapter,
) -> TransformerBackbone:
    return backbone
