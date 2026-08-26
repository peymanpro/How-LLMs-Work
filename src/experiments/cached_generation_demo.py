from __future__ import annotations

from src.inference.cached_generator import (
    CachedTextGenerator,
)
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

    generator = CachedTextGenerator(
        engine
    )

    prompt = [
        0,
        1,
        2,
        3,
    ]

    result = generator.generate(
        prompt=prompt,
        max_new_tokens=3,
    )

    print("HowLLMsWork")
    print("============")
    print()
    print(
        f"Prompt:       {prompt}"
    )
    print(
        f"Generated IDs: {result.token_ids}"
    )
    print(
        f"Cache length: {engine.cache_length}"
    )


if __name__ == "__main__":
    main()
