import numpy as np
import pytest

from src.attention.cached_attention import (
    CachedScaledDotProductAttention,
)
from src.attention.kv_cache import (
    KVCache,
)


def test_cached_attention_should_return_single_position() -> None:
    cache = KVCache(
        key_dimension=2,
        value_dimension=3,
    )

    cache.append(
        np.asarray(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        ),
        np.asarray(
            [
                [2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0],
            ]
        ),
    )

    attention = CachedScaledDotProductAttention(
        key_dimension=2
    )

    result = attention.attend(
        np.asarray(
            [[1.0, 0.0]]
        ),
        cache,
    )

    assert result.shape == (
        1,
        3,
    )


def test_cached_attention_should_match_manual_attention() -> None:
    keys = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    values = np.asarray(
        [
            [2.0, 3.0],
            [5.0, 7.0],
        ]
    )

    query = np.asarray(
        [[1.0, 0.0]]
    )

    cache = KVCache(
        key_dimension=2,
        value_dimension=2,
    )

    cache.append(
        keys,
        values,
    )

    attention = CachedScaledDotProductAttention(
        key_dimension=2
    )

    result = attention.attend(
        query,
        cache,
    )

    scores = (
        query
        @ keys.T
        / np.sqrt(2.0)
    )

    weights = np.exp(
        scores
        - np.max(
            scores
        )
    )

    weights /= np.sum(
        weights,
        axis=1,
        keepdims=True,
    )

    expected = (
        weights
        @ values
    )

    np.testing.assert_allclose(
        result,
        expected,
    )


def test_cached_attention_should_reject_empty_cache() -> None:
    cache = KVCache(
        key_dimension=2,
        value_dimension=2,
    )

    attention = CachedScaledDotProductAttention(
        key_dimension=2
    )

    with pytest.raises(ValueError):
        attention.attend(
            np.ones(
                (1, 2)
            ),
            cache,
        )

from src.attention.scaled_dot_product import (
    ScaledDotProductAttention,
)


def test_cached_attention_should_match_full_attention_for_each_position() -> None:
    rng = np.random.default_rng(
        42
    )

    inputs = rng.normal(
        size=(5, 4)
    )

    full_attention = ScaledDotProductAttention(
        causal=True
    )

    full = full_attention.forward(
        queries=inputs,
        keys=inputs,
        values=inputs,
    )

    cache = KVCache(
        key_dimension=4,
        value_dimension=4,
    )

    cached_attention = (
        CachedScaledDotProductAttention(
            key_dimension=4
        )
    )

    outputs = []

    for position in range(
        inputs.shape[0]
    ):
        cache.append(
            inputs[
                position:position + 1
            ],
            inputs[
                position:position + 1
            ],
        )

        output = cached_attention.attend(
            inputs[
                position:position + 1
            ],
            cache,
        )

        outputs.append(
            output[0]
        )

    cached = np.asarray(
        outputs
    )

    np.testing.assert_allclose(
        cached,
        full.output,
        rtol=1e-12,
        atol=1e-12,
    )
