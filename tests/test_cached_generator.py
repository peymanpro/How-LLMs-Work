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


def create_generator() -> CachedTextGenerator:
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

    return CachedTextGenerator(
        engine
    )


def test_cached_generator_should_append_tokens() -> None:
    generator = create_generator()

    result = generator.generate(
        prompt=[0, 1, 2, 3],
        max_new_tokens=2,
    )

    assert len(
        result.token_ids
    ) == 6

    assert result.token_ids[:4] == (
        0,
        1,
        2,
        3,
    )


def test_cached_generator_should_allow_zero_new_tokens() -> None:
    generator = create_generator()

    result = generator.generate(
        prompt=[0, 1],
        max_new_tokens=0,
    )

    assert result.token_ids == (
        0,
        1,
    )


def test_cached_generator_should_stop_on_eos() -> None:
    from src.inference.sampling_strategy import (
        SamplingStrategy,
    )

    class FixedStrategy(SamplingStrategy):
        def sample(
            self,
            logits,
        ) -> int:
            return 4

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

    generator = CachedTextGenerator(
        engine=PrefillDecodeEngine(
            model
        ),
        strategy=FixedStrategy(),
    )

    result = generator.generate(
        prompt=[0, 1, 2, 3],
        max_new_tokens=5,
        eos_token_id=4,
    )

    assert result.token_ids == (
        0,
        1,
        2,
        3,
        4,
    )
from src.inference.cached_generator import (
    CachedTextGenerator,
)


def test_cached_generator_should_use_incremental_cache() -> None:
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

    result = generator.generate(
        prompt=[0, 1, 2, 3],
        max_new_tokens=2,
    )

    assert len(
        result.token_ids
    ) == 6

    assert engine.cache_length == 6
