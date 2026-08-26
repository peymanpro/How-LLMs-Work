from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PositionalLanguageModelForwardResult:
    logits: np.ndarray
    hidden: np.ndarray


@dataclass(frozen=True)
class PositionalLanguageModelGradients:
    token_embeddings: np.ndarray
    position_embeddings: np.ndarray
    output_weights: np.ndarray
    output_bias: np.ndarray


class PositionalContextLanguageModel:
    def __init__(
        self,
        vocabulary_size: int,
        context_size: int,
        model_dimension: int,
        seed: int = 42,
    ) -> None:
        if vocabulary_size <= 0:
            raise ValueError(
                "Vocabulary size must be positive."
            )

        if context_size <= 0:
            raise ValueError(
                "Context size must be positive."
            )

        if model_dimension <= 0:
            raise ValueError(
                "Model dimension must be positive."
            )

        rng = np.random.default_rng(seed)

        self._vocabulary_size = vocabulary_size
        self._context_size = context_size
        self._model_dimension = model_dimension

        scale = 1.0 / np.sqrt(
            model_dimension
        )

        self._token_embeddings = rng.normal(
            0.0,
            scale,
            size=(
                vocabulary_size,
                model_dimension,
            ),
        )

        self._position_embeddings = rng.normal(
            0.0,
            scale,
            size=(
                context_size,
                model_dimension,
            ),
        )

        self._output_weights = rng.normal(
            0.0,
            scale,
            size=(
                model_dimension,
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
        return self._context_size

    def logits(
        self,
        token_ids: list[int],
    ) -> np.ndarray:
        return self.forward(
            token_ids
        ).logits

    def forward(
        self,
        token_ids: list[int],
    ) -> PositionalLanguageModelForwardResult:
        values = self._validate_tokens(
            token_ids
        )

        hidden = (
            self._token_embeddings[values]
            + self._position_embeddings
        )

        logits = (
            hidden @ self._output_weights
            + self._output_bias
        )

        return PositionalLanguageModelForwardResult(
            logits=logits,
            hidden=hidden,
        )

    def backward(
        self,
        token_ids: list[int],
        forward_result: PositionalLanguageModelForwardResult,
        output_gradient: np.ndarray,
    ) -> PositionalLanguageModelGradients:
        values = self._validate_tokens(
            token_ids
        )

        if output_gradient.shape != forward_result.logits.shape:
            raise ValueError(
                "Output gradient shape does not match logits."
            )

        output_weights_gradient = (
            forward_result.hidden.T
            @ output_gradient
        )

        output_bias_gradient = np.sum(
            output_gradient,
            axis=0,
        )

        hidden_gradient = (
            output_gradient
            @ self._output_weights.T
        )

        token_embeddings_gradient = np.zeros_like(
            self._token_embeddings
        )

        for row, token_id in enumerate(values):
            token_embeddings_gradient[token_id] += (
                hidden_gradient[row]
            )

        position_embeddings_gradient = (
            hidden_gradient
        )

        return PositionalLanguageModelGradients(
            token_embeddings=(
                token_embeddings_gradient
            ),
            position_embeddings=(
                position_embeddings_gradient
            ),
            output_weights=(
                output_weights_gradient
            ),
            output_bias=(
                output_bias_gradient
            ),
        )

    def apply_gradients(
        self,
        gradients: PositionalLanguageModelGradients,
        learning_rate: float,
    ) -> None:
        if not np.isfinite(learning_rate) or learning_rate <= 0.0:
            raise ValueError(
                "Learning rate must be positive and finite."
            )

        self._token_embeddings -= (
            learning_rate
            * gradients.token_embeddings
        )

        self._position_embeddings -= (
            learning_rate
            * gradients.position_embeddings
        )

        self._output_weights -= (
            learning_rate
            * gradients.output_weights
        )

        self._output_bias -= (
            learning_rate
            * gradients.output_bias
        )

    def _validate_tokens(
        self,
        token_ids: list[int],
    ) -> np.ndarray:
        if len(token_ids) != self._context_size:
            raise ValueError(
                "Input sequence length must match context size."
            )

        values = np.asarray(
            token_ids,
            dtype=np.int64,
        )

        if np.any(values < 0) or np.any(
            values >= self._vocabulary_size
        ):
            raise IndexError(
                "Token ID is outside vocabulary."
            )

        return values
