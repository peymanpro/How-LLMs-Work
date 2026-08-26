from __future__ import annotations

import numpy as np

from src.llm.transformer_language_model import (
    TransformerLanguageModel,
)


class DemoBackbone:
    @property
    def model_dimension(self) -> int:
        return 8

    @property
    def context_size(self) -> int:
        return 4

    def forward(
        self,
        token_ids: list[int],
    ) -> np.ndarray:
        hidden = np.zeros(
            (
                len(token_ids),
                self.model_dimension,
            ),
            dtype=np.float64,
        )

        for row, token_id in enumerate(
            token_ids
        ):
            hidden[
                row,
                token_id % self.model_dimension,
            ] = 1.0

        return hidden


def main() -> None:
    model = TransformerLanguageModel(
        backbone=DemoBackbone(),
        vocabulary_size=6,
        seed=42,
    )

    token_ids = [
        0,
        1,
        2,
        3,
    ]

    logits = model.logits(
        token_ids
    )

    print("HowLLMsWork")
    print("============")
    print()
    print("Input token IDs:")
    print(token_ids)
    print()
    print(
        f"Hidden dimension: {model.model_dimension}"
    )
    print(
        f"Vocabulary size:  {model.vocabulary_size}"
    )
    print(
        f"Logits shape:     {logits.shape}"
    )


if __name__ == "__main__":
    main()
