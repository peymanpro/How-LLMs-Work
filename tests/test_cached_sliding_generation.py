from src.inference.cached_generator import (
    CachedTextGenerator,
)
from src.inference.prefill_decode import (
    PrefillDecodeEngine,
)
from src.inference.sampling_strategy import (
    GreedySamplingStrategy,
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
        context_size=4,
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


def test_cached_generator_should_keep_cache_bounded() -> None:
    engine = create_engine()

    generator = CachedTextGenerator(
        engine=engine,
        strategy=GreedySamplingStrategy(),
    )

    result = generator.generate(
        prompt=[0, 1, 2, 3],
        max_new_tokens=4,
    )

    assert len(result.token_ids) == 8
    assert engine.cache_length == 4


def test_cached_generator_should_generate_beyond_context_size() -> None:
    engine = create_engine()

    generator = CachedTextGenerator(
        engine=engine,
        strategy=GreedySamplingStrategy(),
    )

    result = generator.generate(
        prompt=[0, 1, 2, 3],
        max_new_tokens=4,
    )

    assert result.token_ids[:4] == (
        0,
        1,
        2,
        3,
    )

    assert len(result.token_ids) == 8


def test_cached_generator_should_keep_context_window_after_rollover() -> None:
    engine = create_engine()

    generator = CachedTextGenerator(
        engine=engine,
        strategy=GreedySamplingStrategy(),
    )

    generator.generate(
        prompt=[0, 1, 2, 3],
        max_new_tokens=6,
    )

    assert engine.cache_length == 4
