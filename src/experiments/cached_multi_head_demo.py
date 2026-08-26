from __future__ import annotations

import numpy as np

from src.attention.cached_multi_head import (
    CachedMultiHeadAttention,
)
from src.attention.multi_head import (
    TrainableMultiHeadAttention,
)


def main() -> None:
    seed = 42

    cached = CachedMultiHeadAttention(
        model_dimension=8,
        number_of_heads=2,
        seed=seed,
    )

    full = TrainableMultiHeadAttention(
        model_dimension=8,
        number_of_heads=2,
        seed=seed,
        causal=True,
    )

    rng = np.random.default_rng(
        123
    )

    inputs = rng.normal(
        size=(5, 8)
    )

    full_result = full.forward(
        inputs
    )

    print("HowLLMsWork")
    print("============")
    print()
    print(
        "Incremental Multi-Head Attention"
    )
    print()

    for position in range(
        inputs.shape[0]
    ):
        cached_result = (
            cached.forward_token(
                inputs[
                    position:position + 1
                ]
            )
        )

        difference = np.max(
            np.abs(
                cached_result.output[0]
                - full_result.output[position]
            )
        )

        print(
            f"Position {position}: "
            f"cache length={cached.cache_length}, "
            f"max difference={difference:.12f}"
        )


if __name__ == "__main__":
    main()
