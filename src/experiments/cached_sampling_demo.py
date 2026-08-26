from __future__ import annotations

from src.inference.cached_generator import (
    CachedTextGenerator,
)
from src.inference.prefill_decode import (
    PrefillDecodeEngine,
)
from src.inference.sampling_strategy import (
    GreedySamplingStrategy,
    SamplingStrategy,
    TemperatureSamplingStrategy,
    TopKSamplingStrategy,
    TopPSamplingStrategy,
)
from src.llm.cached_transformer_backbone import (
    CachedTransformerBackbone,
)
from src.llm.cached_transformer_language_model import (
    CachedTransformerLanguageModel,
)


def create_engine() -> PrefillDecodeEngine:
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

    return PrefillDecodeEngine(
        model
    )


def main() -> None:
    print("HowLLMsWork")
    print("============")
    print()

    strategies: list[
        tuple[str, SamplingStrategy]
    ] = [
        (
            "Greedy",
            GreedySamplingStrategy(),
        ),
        (
            "Temperature=1.0",
            TemperatureSamplingStrategy(
                temperature=1.0,
                seed=42,
            ),
        ),
        (
            "Top-K=2",
            TopKSamplingStrategy(
                k=2,
                temperature=1.0,
                seed=42,
            ),
        ),
        (
            "Top-P=0.9",
            TopPSamplingStrategy(
                p=0.9,
                temperature=1.0,
                seed=42,
            ),
        ),
    ]

    prompt = [0, 1, 2, 3]

    for name, strategy in strategies:
        engine = create_engine()

        generator = CachedTextGenerator(
            engine=engine,
            strategy=strategy,
        )

        result = generator.generate(
            prompt=prompt,
            max_new_tokens=3,
        )

        print(
            f"{name}:"
        )
        print(
            f"  Generated: {result.token_ids}"
        )
        print(
            f"  Cache:     {engine.cache_length}"
        )
        print()


if __name__ == "__main__":
    main()
