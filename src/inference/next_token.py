from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NextTokenPrediction:
    token_id: int
    probabilities: np.ndarray


class NextTokenPredictor:
    def predict(
        self,
        logits: list[list[float]],
    ) -> NextTokenPrediction:
        values = np.asarray(
            logits,
            dtype=np.float64,
        )

        if values.ndim != 2:
            raise ValueError(
                "Logits must have shape "
                "(sequence_length, vocabulary_size)."
            )

        if values.shape[0] == 0:
            raise ValueError(
                "Logits sequence cannot be empty."
            )

        last_logits = values[-1]

        probabilities = self._softmax(
            last_logits
        )

        token_id = int(
            np.argmax(probabilities)
        )

        return NextTokenPrediction(
            token_id=token_id,
            probabilities=probabilities,
        )

    @staticmethod
    def _softmax(
        logits: np.ndarray,
    ) -> np.ndarray:
        shifted = (
            logits
            - np.max(logits)
        )

        exponentials = np.exp(
            shifted
        )

        denominator = np.sum(
            exponentials
        )

        if (
            not math.isfinite(
                float(denominator)
            )
            or denominator <= 0.0
        ):
            raise ValueError(
                "Cannot normalize logits into probabilities."
            )

        return (
            exponentials
            / denominator
        )
