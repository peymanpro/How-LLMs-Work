from __future__ import annotations

import numpy as np

from src.training.evaluation import (
    LanguageModelEvaluator,
)


def main() -> None:
    logits = np.array(
        [
            [
                [4.0, 1.0, 0.0],
                [0.0, 4.0, 1.0],
                [1.0, 0.0, 4.0],
            ]
        ],
        dtype=np.float64,
    )

    targets = np.array(
        [
            [0, 1, 2],
        ],
        dtype=np.int64,
    )

    result = LanguageModelEvaluator().evaluate(
        logits,
        targets,
    )

    print("HowLLMsWork")
    print("============")
    print()
    print(
        f"Loss:       {result.loss:.6f}"
    )
    print(
        f"Perplexity: {result.perplexity:.6f}"
    )


if __name__ == "__main__":
    main()
