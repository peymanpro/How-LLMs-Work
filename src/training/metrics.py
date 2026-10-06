from __future__ import annotations

import numpy as np


class LanguageModelMetrics:
    @staticmethod
    def token_accuracy(
        logits: np.ndarray,
        targets: np.ndarray,
    ) -> float:
        if logits.ndim != 3:
            raise ValueError(
                "Logits must have shape "
                "(batch_size, sequence_length, vocabulary_size)."
            )

        if targets.ndim != 2:
            raise ValueError(
                "Targets must have shape "
                "(batch_size, sequence_length)."
            )

        if logits.shape[:2] != targets.shape:
            raise ValueError(
                "Logits and targets dimensions do not match."
            )

        if targets.size == 0:
            raise ValueError(
                "Targets cannot be empty."
            )

        if np.any(targets < 0) or np.any(
            targets >= logits.shape[2]
        ):
            raise ValueError(
                "Targets contain token IDs outside the vocabulary."
            )

        predictions = np.argmax(
            logits,
            axis=2,
        )

        return float(
            np.mean(
                predictions == targets
            )
        )
