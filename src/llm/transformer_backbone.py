from __future__ import annotations

from typing import Protocol

import numpy as np


class TransformerBackbone(Protocol):
    @property
    def model_dimension(self) -> int:
        ...

    @property
    def context_size(self) -> int:
        ...

    def forward(
        self,
        token_ids: list[int],
    ) -> np.ndarray:
        """
        Return one hidden representation per input token.

        Shape:
            (sequence_length, model_dimension)
        """
        ...
