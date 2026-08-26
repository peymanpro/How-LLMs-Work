import numpy as np
import pytest

from src.llm.cached_transformer_backbone import (
    CachedTransformerBackbone,
)
from src.llm.cached_transformer_language_model import (
    CachedTransformerLanguageModel,
)


def create_model() -> CachedTransformerLanguageModel:
    backbone = CachedTransformerBackbone(
        vocabulary_size=5,
        model_dimension=8,
        number_of_heads=2,
        context_size=4,
        seed=42,
    )

    return CachedTransformerLanguageModel(
        backbone=backbone,
        vocabulary_size=5,
        seed=42,
    )


def test_cached_language_model_should_return_expected_logits_shape() -> None:
    model = create_model()

    result = model.logits(
        [0, 1, 2, 3]
    )

    assert result.shape == (
        4,
        5,
    )


def test_incremental_logits_should_return_one_vocabulary_vector() -> None:
    model = create_model()

    result = model.incremental_logits(
        0
    )

    assert result.shape == (
        5,
    )


def test_incremental_logits_should_update_cache() -> None:
    model = create_model()

    assert model.cache_length == 0

    model.incremental_logits(0)

    assert model.cache_length == 1

    model.incremental_logits(1)

    assert model.cache_length == 2


def test_reset_cache_should_reset_incremental_state() -> None:
    model = create_model()

    model.incremental_logits(0)
    model.incremental_logits(1)

    model.reset_cache()

    assert model.cache_length == 0


def test_forward_should_reset_previous_cache_state() -> None:
    model = create_model()

    model.incremental_logits(0)

    first = model.logits(
        [0, 1]
    )

    second = model.logits(
        [0, 1]
    )

    np.testing.assert_allclose(
        first,
        second,
    )


def test_incremental_logits_should_match_last_forward_logits() -> None:
    model = create_model()

    token_ids = [0, 1, 2, 3]

    full = model.logits(
        token_ids
    )

    model.reset_cache()

    for token_id in token_ids[:-1]:
        model.incremental_logits(
            token_id
        )

    incremental = model.incremental_logits(
        token_ids[-1]
    )

    np.testing.assert_allclose(
        incremental,
        full[-1],
        rtol=1e-12,
        atol=1e-12,
    )


def test_model_should_reject_invalid_token() -> None:
    model = create_model()

    with pytest.raises(ValueError):
        model.incremental_logits(5)
