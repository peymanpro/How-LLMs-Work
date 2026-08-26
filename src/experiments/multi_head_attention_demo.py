from __future__ import annotations

import numpy as np

from src.attention.multi_head import (
    TrainableMultiHeadAttention,
)


def main() -> None:
    attention = TrainableMultiHeadAttention(
        model_dimension=8,
        number_of_heads=2,
        seed=42,
        causal=True,
    )

    inputs = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )

    result = attention.forward(
        inputs
    )

    print("HowLLMsWork")
    print("============")
    print()
    print(
        f"Model dimension: {attention.model_dimension}"
    )
    print(
        f"Number of heads: {attention.number_of_heads}"
    )
    print(
        f"Head dimension:  {attention.head_dimension}"
    )
    print()

    for index, head in enumerate(
        result.heads,
    ):
        print(
            f"Head {index + 1} attention weights:"
        )
        print(
            np.array2string(
                head.weights,
                precision=4,
            )
        )
        print()

    print("Concatenated shape:")
    print(
        result.concatenated.shape
    )
    print()

    print("Final output shape:")
    print(
        result.output.shape
    )


if __name__ == "__main__":
    main()
