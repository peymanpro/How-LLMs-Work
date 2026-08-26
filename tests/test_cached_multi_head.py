import numpy as np
import pytest

from src.attention.cached_multi_head import (
    CachedMultiHeadAttention,
)
from src.attention.multi_head import (
    TrainableMultiHeadAttention,
)


def create_cached_attention() -> CachedMultiHeadAttention:
    return CachedMultiHeadAttention(
        model_dimension=8,
        number_of_heads=2,
        seed=42,
    )


def test_cached_multi_head_should_return_single_token_output() -> None:
    attention = create_cached_attention()

    result = attention.forward_token(
        np.ones(
            (1, 8)
        )
    )

    assert result.query.shape == (
        1,
        8,
    )

    assert result.key.shape == (
        1,
        8,
    )

    assert result.value.shape == (
        1,
        8,
    )

    assert len(
        result.head_outputs
    ) == 2

    assert result.head_outputs[0].shape == (
        1,
        4,
    )

    assert result.output.shape == (
        1,
        8,
    )


def test_cached_multi_head_should_increase_cache_length() -> None:
    attention = create_cached_attention()

    for position in range(4):
        attention.forward_token(
            np.ones(
                (
                    1,
                    8,
                )
            )
        )

        assert attention.cache_length == (
            position + 1
        )


def test_cached_multi_head_should_reset_cache() -> None:
    attention = create_cached_attention()

    attention.forward_token(
        np.ones(
            (1, 8)
        )
    )

    assert attention.cache_length == 1

    attention.reset_cache()

    assert attention.cache_length == 0


def test_cached_multi_head_should_reject_multiple_tokens() -> None:
    attention = create_cached_attention()

    with pytest.raises(ValueError):
        attention.forward_token(
            np.ones(
                (2, 8)
            )
        )


def test_cached_multi_head_should_reject_wrong_dimension() -> None:
    attention = create_cached_attention()

    with pytest.raises(ValueError):
        attention.forward_token(
            np.ones(
                (1, 7)
            )
        )


def test_cached_multi_head_should_produce_finite_output() -> None:
    attention = create_cached_attention()

    rng = np.random.default_rng(
        42
    )

    for _ in range(5):
        result = attention.forward_token(
            rng.normal(
                size=(1, 8)
            )
        )

        assert np.isfinite(
            result.output
        ).all()


def test_cached_multi_head_should_match_single_step_full_attention() -> None:
    seed = 42

    cached = CachedMultiHeadAttention(
        model_dimension=8,
        number_of_heads=2,
        seed=seed,
    )

    full = TrainableMultiHeadAttention(
        model_dimension=8,
        number_of_heads=2,
        seed=seed,
        causal=True,
    )

    rng = np.random.default_rng(
        123
    )

    inputs = rng.normal(
        size=(1, 8)
    )

    cached_result = cached.forward_token(
        inputs
    )

    full_result = full.forward(
        inputs
    )

    np.testing.assert_allclose(
        cached_result.output,
        full_result.output,
        rtol=1e-12,
        atol=1e-12,
    )

from src.attention.cached_multi_head import (
    CachedMultiHeadAttention,
)


def test_incremental_multi_head_attention_should_match_full_attention() -> None:
    seed = 42

    cached = CachedMultiHeadAttention(
        model_dimension=8,
        number_of_heads=2,
        seed=seed,
    )

    full = TrainableMultiHeadAttention(
        model_dimension=8,
        number_of_heads=2,
        seed=seed,
        causal=True,
    )

    rng = np.random.default_rng(
        123
    )

    inputs = rng.normal(
        size=(5, 8)
    )

    full_result = full.forward(
        inputs
    )

    incremental_outputs = []

    for position in range(
        inputs.shape[0]
    ):
        result = cached.forward_token(
            inputs[
                position:position + 1
            ]
        )

        incremental_outputs.append(
            result.output[0]
        )

    incremental = np.asarray(
        incremental_outputs
    )

    np.testing.assert_allclose(
        incremental,
        full_result.output,
        rtol=1e-12,
        atol=1e-12,
    )
