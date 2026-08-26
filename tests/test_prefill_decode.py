import pytest

from src.inference.prefill_decode import (
    PrefillDecodeEngine,
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


def test_prefill_should_populate_cache() -> None:
    engine = create_engine()

    result = engine.prefill(
        [0, 1, 2, 3]
    )

    assert result.token_ids == (
        0,
        1,
        2,
        3,
    )

    assert len(
        result.next_logits
    ) == 5

    assert result.cache_length == 4
    assert engine.cache_length == 4


def test_decode_should_append_one_token() -> None:
    engine = create_engine()

    engine.prefill(
        [0, 1, 2, 3]
    )

    result = engine.decode(
        4
    )

    assert result.token_id == 4
    assert len(
        result.next_logits
    ) == 5

    assert result.cache_length == 5


def test_prefill_should_reset_previous_cache() -> None:
    engine = create_engine()

    engine.prefill(
        [0, 1, 2]
    )

    assert engine.cache_length == 3

    result = engine.prefill(
        [3, 4]
    )

    assert result.cache_length == 2
    assert engine.cache_length == 2


def test_reset_should_clear_cache() -> None:
    engine = create_engine()

    engine.prefill(
        [0, 1, 2]
    )

    engine.reset()

    assert engine.cache_length == 0


def test_prefill_should_reject_empty_prompt() -> None:
    engine = create_engine()

    with pytest.raises(ValueError):
        engine.prefill([])


def test_prefill_logits_should_match_sequential_incremental_logits() -> None:
    engine = create_engine()

    prompt = [
        0,
        1,
        2,
        3,
    ]

    prefetched = engine.prefill(
        prompt
    )

    engine.reset()

    logits = []

    for token_id in prompt:
        logits = engine.decode(
            token_id
        ).next_logits

    assert prefetched.next_logits == pytest.approx(
        logits
    )


def test_prefill_then_decode_should_match_full_forward() -> None:
    engine = create_engine()

    prompt = [
        0,
        1,
        2,
        3,
    ]

    engine.prefill(
        prompt
    )

    decoded = engine.decode(
        4
    )

    engine.reset()

    full_logits = engine._model.logits(
        prompt + [4]
    )

    assert decoded.next_logits == pytest.approx(
        full_logits[-1].tolist()
    )
