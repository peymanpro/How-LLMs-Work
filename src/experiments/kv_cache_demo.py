from __future__ import annotations

import numpy as np

from src.attention.cached_attention import (
    CachedScaledDotProductAttention,
)
from src.attention.kv_cache import (
    KVCache,
)
from src.attention.scaled_dot_product import (
    ScaledDotProductAttention,
)


def main() -> None:
    rng = np.random.default_rng(
        42
    )

    inputs = rng.normal(
        size=(4, 4)
    )

    full_attention = ScaledDotProductAttention(
        causal=True
    )

    full = full_attention.forward(
        inputs,
        inputs,
        inputs,
    )

    cache = KVCache(
        key_dimension=4,
        value_dimension=4,
    )

    cached_attention = (
        CachedScaledDotProductAttention(
            key_dimension=4
        )
    )

    print("HowLLMsWork")
    print("============")
    print()
    print(
        "Incremental decoding:"
    )

    for position in range(
        inputs.shape[0]
    ):
        cache.append(
            inputs[
                position:position + 1
            ],
            inputs[
                position:position + 1
            ],
        )

        output = cached_attention.attend(
            inputs[
                position:position + 1
            ],
            cache,
        )

        difference = np.max(
            np.abs(
                output[0]
                - full.output[position]
            )
        )

        print(
            f"Position {position}: "
            f"cache length={cache.sequence_length}, "
            f"max difference={difference:.12f}"
        )


if __name__ == "__main__":
    main()
