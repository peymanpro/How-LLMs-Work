from __future__ import annotations

from src.llm.cached_transformer_backbone import (
    CachedTransformerBackbone,
)


def main() -> None:
    backbone = CachedTransformerBackbone(
        vocabulary_size=5,
        model_dimension=8,
        number_of_heads=2,
        context_size=4,
        seed=42,
    )

    token_ids = [
        0,
        1,
        2,
        3,
    ]

    print("HowLLMsWork")
    print("============")
    print()

    print(
        "Incremental inference:"
    )

    for token_id in token_ids:
        hidden = backbone.incremental_forward(
            token_id
        )

        print(
            f"token={token_id} "
            f"position={backbone.position} "
            f"cache={backbone.cache_length} "
            f"shape={hidden.shape}"
        )

    backbone.reset_cache()

    full = backbone.forward(
        token_ids
    )

    print()
    print(
        f"Full output shape: {full.shape}"
    )
    print(
        f"Final cache length: {backbone.cache_length}"
    )


if __name__ == "__main__":
    main()
