import numpy as np

from src.inference.cached_generator import (
    CachedTextGenerator,
)
from src.inference.prefill_decode import (
    PrefillDecodeEngine,
)
from src.inference.sampling_strategy import (
    GreedySamplingStrategy,
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


def test_cached_generation_should_support_greedy_strategy() -> None:
    engine = create_engine()

    generator = CachedTextGenerator(
        engine=engine,
        strategy=GreedySamplingStrategy(),
    )

    result = generator.generate(
        prompt=[0, 1, 2, 3],
        max_new_tokens=2,
    )

    assert len(result.token_ids) == 6
    assert engine.cache_length == 6


def test_cached_generation_should_support_temperature_strategy() -> None:
    engine = create_engine()

    generator = CachedTextGenerator(
        engine=engine,
        strategy=TemperatureSamplingStrategy(
            temperature=1.0,
            seed=42,
        ),
    )

    result = generator.generate(
        prompt=[0, 1, 2, 3],
        max_new_tokens=2,
    )

    assert len(result.token_ids) == 6
    assert all(
        0 <= token_id < 5
        for token_id in result.token_ids
    )
    assert engine.cache_length == 6


def test_cached_generation_should_support_top_k_strategy() -> None:
    engine = create_engine()

    generator = CachedTextGenerator(
        engine=engine,
        strategy=TopKSamplingStrategy(
            k=2,
            temperature=1.0,
            seed=42,
        ),
    )

    result = generator.generate(
        prompt=[0, 1, 2, 3],
        max_new_tokens=2,
    )

    assert len(result.token_ids) == 6
    assert engine.cache_length == 6


def test_cached_generation_should_support_top_p_strategy() -> None:
    engine = create_engine()

    generator = CachedTextGenerator(
        engine=engine,
        strategy=TopPSamplingStrategy(
            p=0.9,
            temperature=1.0,
            seed=42,
        ),
    )

    result = generator.generate(
        prompt=[0, 1, 2, 3],
        max_new_tokens=2,
    )

    assert len(result.token_ids) == 6
    assert engine.cache_length == 6


def test_cached_sampling_should_keep_prompt_unchanged() -> None:
    engine = create_engine()

    generator = CachedTextGenerator(
        engine=engine,
        strategy=TemperatureSamplingStrategy(
            temperature=2.0,
            seed=42,
        ),
    )

    prompt = [0, 1, 2, 3]

    result = generator.generate(
        prompt=prompt,
        max_new_tokens=2,
    )

    assert result.token_ids[:4] == tuple(
        prompt
    )


def test_cached_sampling_should_stop_on_eos() -> None:
    class FixedEosStrategy:
        def sample(
            self,
            logits: np.ndarray,
        ) -> int:
            return 4

    engine = create_engine()

    generator = CachedTextGenerator(
        engine=engine,
        strategy=FixedEosStrategy(),
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

    assert engine.cache_length == 4
