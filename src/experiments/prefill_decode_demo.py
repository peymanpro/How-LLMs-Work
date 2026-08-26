from __future__ import annotations

from src.inference.prefill_decode import (
    PrefillDecodeEngine,
)
from src.llm.cached_transformer_backbone import (
    CachedTransformerBackbone,
)
from src.llm.cached_transformer_language_model import (
    CachedTransformerLanguageModel,
)


def main() -> None:
    backbone = CachedTransformerBackbone(
        vocabulary_size=5,
        model_dimension=8,
        number_of_heads=2,
        context_size=8,
        seed=42,
    )

    model = CachedTransformerLanguageModel(
        backbone=backbone,
        vocabulary_size=5,
        seed=42,
    )

    engine = PrefillDecodeEngine(
        model
    )

    prompt = [
        0,
        1,
        2,
        3,
    ]

    print("HowLLMsWork")
    print("============")
    print()
    print("Prefill")
    print("-------")

    prefill = engine.prefill(
        prompt
    )

    print(
        f"Prompt:       {prefill.token_ids}"
    )

    print(
        f"Cache length: {prefill.cache_length}"
    )

    print(
        f"Logits size:  {len(prefill.next_logits)}"
    )

    print()
    print("Decode")
    print("------")

    token_id = 4

    decoded = engine.decode(
        token_id
    )

    print(
        f"Input token:  {decoded.token_id}"
    )

    print(
        f"Cache length: {decoded.cache_length}"
    )

    print(
        f"Logits size:  {len(decoded.next_logits)}"
    )


if __name__ == "__main__":
    main()
