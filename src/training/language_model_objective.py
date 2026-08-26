from __future__ import annotations

import math

import numpy as np


class LanguageModelObjective:
    def cross_entropy(
        self,
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

        shifted = (
            logits
            - np.max(
                logits,
                axis=2,
                keepdims=True,
            )
        )

        log_sum_exp = np.log(
            np.sum(
                np.exp(shifted),
                axis=2,
            )
        )

        target_logits = np.take_along_axis(
            shifted,
            targets[..., None],
            axis=2,
        ).squeeze(
            axis=2
        )

        losses = (
            -target_logits
            + log_sum_exp
        )

        return float(
            np.mean(losses)
        )

    def perplexity(
        self,
        loss: float,
    ) -> float:
        if not math.isfinite(loss):
            raise ValueError(
                "Loss must be finite."
            )

        if loss < 0.0:
            raise ValueError(
                "Loss cannot be negative."
            )

        return math.exp(loss)
