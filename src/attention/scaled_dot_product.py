from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AttentionResult:
    queries: np.ndarray
    keys: np.ndarray
    values: np.ndarray
    scores: np.ndarray
    weights: np.ndarray
    output: np.ndarray


class ScaledDotProductAttention:
    def __init__(
        self,
        causal: bool = False,
    ) -> None:
        self._causal = causal

    def forward(
        self,
        queries: np.ndarray,
        keys: np.ndarray,
        values: np.ndarray,
    ) -> AttentionResult:
        q = np.asarray(
            queries,
            dtype=np.float64,
        )

        k = np.asarray(
            keys,
            dtype=np.float64,
        )

        v = np.asarray(
            values,
            dtype=np.float64,
        )

        if q.ndim != 2:
            raise ValueError(
                "Queries must be a two-dimensional matrix."
            )

        if k.ndim != 2:
            raise ValueError(
                "Keys must be a two-dimensional matrix."
            )

        if v.ndim != 2:
            raise ValueError(
                "Values must be a two-dimensional matrix."
            )

        if q.shape[1] != k.shape[1]:
            raise ValueError(
                "Query and key dimensions must match."
            )

        if k.shape[0] != v.shape[0]:
            raise ValueError(
                "Key and value sequence lengths must match."
            )

        if q.shape[0] == 0:
            raise ValueError(
                "Sequence cannot be empty."
            )

        head_dimension = q.shape[1]

        scores = (
            q
            @ k.T
            / np.sqrt(head_dimension)
        )

        if self._causal:
            if q.shape[0] != k.shape[0]:
                raise ValueError(
                    "Causal attention requires equal query "
                    "and key sequence lengths."
                )

            mask = np.triu(
                np.ones_like(
                    scores,
                    dtype=bool,
                ),
                k=1,
            )

            scores = np.where(
                mask,
                -np.inf,
                scores,
            )

        weights = self._softmax(
            scores
        )

        output = weights @ v

        return AttentionResult(
            queries=q,
            keys=k,
            values=v,
            scores=scores,
            weights=weights,
            output=output,
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

        shifted = (
            values
            - maximum
        )

        exponentials = np.exp(
            shifted
        )

        totals = np.sum(
            exponentials,
            axis=1,
            keepdims=True,
        )

        return (
            exponentials
            / totals
        )
