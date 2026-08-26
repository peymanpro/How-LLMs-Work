from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LanguageModelForwardResult:
    logits: np.ndarray
    hidden: np.ndarray


@dataclass(frozen=True)
class LanguageModelGradients:
    embeddings: np.ndarray
    output_weights: np.ndarray
    output_bias: np.ndarray


class SimpleContextLanguageModel:
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

        self._embeddings = (
            rng.normal(
                0.0,
                scale,
                size=(
                    vocabulary_size,
                    model_dimension,
                ),
            )
        )

        self._output_weights = (
            rng.normal(
                0.0,
                scale,
                size=(
                    model_dimension,
                    vocabulary_size,
                ),
            )
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

    @property
    def model_dimension(self) -> int:
        return self._model_dimension

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
    ) -> LanguageModelForwardResult:
        token_array = self._validate_tokens(
            token_ids
        )

        hidden = np.mean(
            self._embeddings[token_array],
            axis=0,
        )

        logits = (
            hidden @ self._output_weights
            + self._output_bias
        )

        return LanguageModelForwardResult(
            logits=np.broadcast_to(
                logits,
                (
                    self._context_size,
                    self._vocabulary_size,
                ),
            ).copy(),
            hidden=hidden,
        )

    def backward(
        self,
        token_ids: list[int],
        forward_result: LanguageModelForwardResult,
        output_gradient: np.ndarray,
    ) -> LanguageModelGradients:
        token_array = self._validate_tokens(
            token_ids
        )

        if output_gradient.shape != forward_result.logits.shape:
            raise ValueError(
                "Output gradient shape does not match logits."
            )

        effective_gradient = np.sum(
            output_gradient,
            axis=0,
        )

        output_weights_gradient = np.outer(
            forward_result.hidden,
            effective_gradient,
        )

        output_bias_gradient = (
            effective_gradient
        )

        hidden_gradient = (
            self._output_weights
            @ effective_gradient
        )

        embedding_gradient = np.zeros_like(
            self._embeddings
        )

        per_token_gradient = (
            hidden_gradient
            / len(token_array)
        )

        for token_id in token_array:
            embedding_gradient[token_id] += (
                per_token_gradient
            )

        return LanguageModelGradients(
            embeddings=embedding_gradient,
            output_weights=output_weights_gradient,
            output_bias=output_bias_gradient,
        )

    def apply_gradients(
        self,
        gradients: LanguageModelGradients,
        learning_rate: float,
    ) -> None:
        if not np.isfinite(learning_rate) or learning_rate <= 0.0:
            raise ValueError(
                "Learning rate must be positive and finite."
            )

        if gradients.embeddings.shape != self._embeddings.shape:
            raise ValueError(
                "Embedding gradient has incorrect shape."
            )

        if (
            gradients.output_weights.shape
            != self._output_weights.shape
        ):
            raise ValueError(
                "Output weight gradient has incorrect shape."
            )

        if (
            gradients.output_bias.shape
            != self._output_bias.shape
        ):
            raise ValueError(
                "Output bias gradient has incorrect shape."
            )

        self._embeddings -= (
            learning_rate
            * gradients.embeddings
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

