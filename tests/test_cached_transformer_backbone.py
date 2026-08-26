import numpy as np
import pytest

from src.llm.cached_transformer_backbone import (
    CachedTransformerBackbone,
)


def create_backbone() -> CachedTransformerBackbone:
    return CachedTransformerBackbone(
        vocabulary_size=5,
        model_dimension=8,
        number_of_heads=2,
        context_size=4,
        seed=42,
    )


def test_cached_backbone_should_return_expected_shape() -> None:
    backbone = create_backbone()

    result = backbone.forward(
        [0, 1, 2, 3]
    )

    assert result.shape == (
        4,
        8,
    )


def test_cached_backbone_should_update_cache_incrementally() -> None:
    backbone = create_backbone()

    for index in range(4):
        result = backbone.incremental_forward(
            index
        )

        assert result.shape == (
            1,
            8,
        )

        assert backbone.position == index + 1
        assert backbone.cache_length == index + 1


def test_cached_backbone_should_reset_cache() -> None:
    backbone = create_backbone()

    backbone.incremental_forward(0)
    backbone.incremental_forward(1)

    backbone.reset_cache()

    assert backbone.position == 0
    assert backbone.cache_length == 0


def test_cached_backbone_forward_should_reset_previous_state() -> None:
    backbone = create_backbone()

    backbone.incremental_forward(0)

    first = backbone.forward(
        [0, 1]
    )

    second = backbone.forward(
        [0, 1]
    )

    np.testing.assert_allclose(
        first,
        second,
    )


def test_cached_backbone_should_reject_invalid_token() -> None:
    backbone = create_backbone()

    with pytest.raises(ValueError):
        backbone.incremental_forward(5)


def test_cached_backbone_should_reject_context_overflow() -> None:
    backbone = create_backbone()

    for token_id in [0, 1, 2, 3]:
        backbone.incremental_forward(
            token_id
        )

    with pytest.raises(ValueError):
        backbone.incremental_forward(4)


def test_incremental_outputs_should_match_forward_outputs() -> None:
    backbone = create_backbone()

    token_ids = [0, 1, 2, 3]

    full = backbone.forward(
        token_ids
    )

    backbone.reset_cache()

    incremental = np.concatenate(
        [
            backbone.incremental_forward(
                token_id
            )
            for token_id in token_ids
        ],
        axis=0,
    )

    np.testing.assert_allclose(
        incremental,
        full,
        rtol=1e-12,
        atol=1e-12,
    )
