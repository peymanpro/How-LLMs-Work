from __future__ import annotations

import numpy as np

from src.attention.scaled_dot_product import (
    ScaledDotProductAttention,
)


def main() -> None:
    inputs = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    attention = ScaledDotProductAttention(
        causal=True
    )

    result = attention.forward(
        queries=inputs,
        keys=inputs,
        values=inputs,
    )

    print("HowLLMsWork")
    print("============")
    print()
    print("Attention weights:")
    print(
        np.array2string(
            result.weights,
            precision=4,
        )
    )
    print()
    print("Output:")
    print(
        np.array2string(
            result.output,
            precision=4,
        )
    )


if __name__ == "__main__":
    main()
