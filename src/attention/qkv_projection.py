from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class QKVProjectionResult:
    inputs: np.ndarray
    queries: np.ndarray
    keys: np.ndarray
    values: np.ndarray


@dataclass(frozen=True)
class QKVGradients:
    weights_q: np.ndarray
    weights_k: np.ndarray
    weights_v: np.ndarray
    input: np.ndarray


class TrainableQKVProjection:
    def __init__(
        self,
        input_dimension: int,
        attention_dimension: int,
        seed: int = 42,
    ) -> None:
        if input_dimension <= 0:
            raise ValueError(
                "input_dimension must be positive."
            )

        if attention_dimension <= 0:
            raise ValueError(
                "attention_dimension must be positive."
            )

        self._input_dimension = input_dimension
        self._attention_dimension = (
            attention_dimension
        )

        rng = np.random.default_rng(
            seed
        )

        scale = 1.0 / np.sqrt(
            input_dimension
        )

        self._weights_q = rng.normal(
            0.0,
            scale,
            size=(
                input_dimension,
                attention_dimension,
            ),
        )

        self._weights_k = rng.normal(
            0.0,
            scale,
            size=(
                input_dimension,
                attention_dimension,
            ),
        )

        self._weights_v = rng.normal(
            0.0,
            scale,
            size=(
                input_dimension,
                attention_dimension,
            ),
        )

    @property
    def input_dimension(self) -> int:
        return self._input_dimension

    @property
    def attention_dimension(self) -> int:
        return self._attention_dimension

    @property
    def weights_q(self) -> np.ndarray:
        return self._weights_q.copy()

    @property
    def weights_k(self) -> np.ndarray:
        return self._weights_k.copy()

    @property
    def weights_v(self) -> np.ndarray:
        return self._weights_v.copy()

    def forward(
        self,
        inputs: np.ndarray,
    ) -> QKVProjectionResult:
        values = np.asarray(
            inputs,
            dtype=np.float64,
        )

        if values.ndim != 2:
            raise ValueError(
                "Inputs must be a two-dimensional matrix."
            )

        if values.shape[1] != self._input_dimension:
            raise ValueError(
                "Input dimension does not match projection."
            )

        return QKVProjectionResult(
            inputs=values,
            queries=(
                values
                @ self._weights_q
            ),
            keys=(
                values
                @ self._weights_k
            ),
            values=(
                values
                @ self._weights_v
            ),
        )

    def backward(
        self,
        forward_result: QKVProjectionResult,
        query_gradient: np.ndarray,
        key_gradient: np.ndarray,
        value_gradient: np.ndarray,
    ) -> QKVGradients:
        d_q = np.asarray(
            query_gradient,
            dtype=np.float64,
        )

        d_k = np.asarray(
            key_gradient,
            dtype=np.float64,
        )

        d_v = np.asarray(
            value_gradient,
            dtype=np.float64,
        )

        expected = (
            forward_result.queries.shape
        )

        if d_q.shape != expected:
            raise ValueError(
                "Query gradient shape does not match queries."
            )

        if d_k.shape != (
            forward_result.keys.shape
        ):
            raise ValueError(
                "Key gradient shape does not match keys."
            )

        if d_v.shape != (
            forward_result.values.shape
        ):
            raise ValueError(
                "Value gradient shape does not match values."
            )

        inputs = forward_result.inputs

        d_weights_q = (
            inputs.T
            @ d_q
        )

        d_weights_k = (
            inputs.T
            @ d_k
        )

        d_weights_v = (
            inputs.T
            @ d_v
        )

        d_inputs = (
            d_q @ self._weights_q.T
            + d_k @ self._weights_k.T
            + d_v @ self._weights_v.T
        )

        return QKVGradients(
            weights_q=d_weights_q,
            weights_k=d_weights_k,
            weights_v=d_weights_v,
            input=d_inputs,
        )

    def apply_gradients(
        self,
        gradients: QKVGradients,
        learning_rate: float,
    ) -> None:
        if not np.isfinite(
            learning_rate
        ) or learning_rate <= 0.0:
            raise ValueError(
                "learning_rate must be positive and finite."
            )

        if gradients.weights_q.shape != (
            self._weights_q.shape
        ):
            raise ValueError(
                "Q gradient shape does not match weights."
            )

        if gradients.weights_k.shape != (
            self._weights_k.shape
        ):
            raise ValueError(
                "K gradient shape does not match weights."
            )

        if gradients.weights_v.shape != (
            self._weights_v.shape
        ):
            raise ValueError(
                "V gradient shape does not match weights."
            )

        self._weights_q = (
            self._weights_q
            - learning_rate
            * gradients.weights_q
        )

        self._weights_k = (
            self._weights_k
            - learning_rate
            * gradients.weights_k
        )

        self._weights_v = (
            self._weights_v
            - learning_rate
            * gradients.weights_v
        )
