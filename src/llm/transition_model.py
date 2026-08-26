from __future__ import annotations

import numpy as np


class TransitionTableLanguageModel:
    def __init__(
        self,
        vocabulary_size: int,
        context_size: int,
        transitions: dict[int, int],
    ) -> None:
        if vocabulary_size <= 0:
            raise ValueError(
                "Vocabulary size must be positive."
            )

        if context_size <= 0:
            raise ValueError(
                "Context size must be positive."
            )

        for source, target in transitions.items():
            if not 0 <= source < vocabulary_size:
                raise ValueError(
                    "Transition source is outside vocabulary."
                )

            if not 0 <= target < vocabulary_size:
                raise ValueError(
                    "Transition target is outside vocabulary."
                )

        self._vocabulary_size = vocabulary_size
        self._context_size = context_size
        self._transitions = dict(transitions)

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
        if len(token_ids) != self._context_size:
            raise ValueError(
                "Input sequence length does not "
                "match context size."
            )

        logits = np.zeros(
            (
                self._context_size,
                self._vocabulary_size,
            ),
            dtype=np.float64,
        )

        for row, token_id in enumerate(token_ids):
            if not 0 <= token_id < self._vocabulary_size:
                raise IndexError(
                    "Token ID is outside vocabulary."
                )

            target = self._transitions.get(token_id)

            if target is not None:
                logits[row, target] = 10.0

        return logits
