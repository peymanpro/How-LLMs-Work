from __future__ import annotations

from typing import Protocol

import numpy as np


class LanguageModel(Protocol):
    @property
    def vocabulary_size(self) -> int:
        ...

    @property
    def context_size(self) -> int:
        ...

    def logits(
        self,
        token_ids: list[int],
    ) -> np.ndarray:
        """
        Return one vocabulary-logit vector per input token.

        Shape:
            (sequence_length, vocabulary_size)
        """
        ...
