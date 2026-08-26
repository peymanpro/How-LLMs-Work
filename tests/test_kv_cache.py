import numpy as np
import pytest

from src.attention.kv_cache import (
    KVCache,
)


def test_kv_cache_should_start_empty() -> None:
    cache = KVCache(
        key_dimension=4,
        value_dimension=6,
    )

    assert cache.sequence_length == 0
    assert cache.state.keys.shape == (
        0,
        4,
    )
    assert cache.state.values.shape == (
        0,
        6,
    )


def test_kv_cache_should_append_keys_and_values() -> None:
    cache = KVCache(
        key_dimension=2,
        value_dimension=3,
    )

    keys = np.asarray(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )

    values = np.asarray(
        [
            [5.0, 6.0, 7.0],
            [8.0, 9.0, 10.0],
        ]
    )

    cache.append(
        keys,
        values,
    )

    assert cache.sequence_length == 2

    np.testing.assert_allclose(
        cache.state.keys,
        keys,
    )

    np.testing.assert_allclose(
        cache.state.values,
        values,
    )


def test_kv_cache_should_accumulate_multiple_appends() -> None:
    cache = KVCache(
        key_dimension=2,
        value_dimension=2,
    )

    cache.append(
        np.asarray(
            [[1.0, 2.0]]
        ),
        np.asarray(
            [[3.0, 4.0]]
        ),
    )

    cache.append(
        np.asarray(
            [[5.0, 6.0]]
        ),
        np.asarray(
            [[7.0, 8.0]]
        ),
    )

    assert cache.sequence_length == 2

    np.testing.assert_allclose(
        cache.state.keys,
        [
            [1.0, 2.0],
            [5.0, 6.0],
        ],
    )


def test_kv_cache_should_clear() -> None:
    cache = KVCache(
        key_dimension=2,
        value_dimension=2,
    )

    cache.append(
        np.ones(
            (3, 2)
        ),
        np.ones(
            (3, 2)
        ),
    )

    cache.clear()

    assert cache.sequence_length == 0


def test_kv_cache_should_reject_wrong_key_dimension() -> None:
    cache = KVCache(
        key_dimension=2,
        value_dimension=2,
    )

    with pytest.raises(ValueError):
        cache.append(
            np.ones(
                (1, 3)
            ),
            np.ones(
                (1, 2)
            ),
        )


def test_kv_cache_should_reject_mismatched_lengths() -> None:
    cache = KVCache(
        key_dimension=2,
        value_dimension=2,
    )

    with pytest.raises(ValueError):
        cache.append(
            np.ones(
                (2, 2)
            ),
            np.ones(
                (1, 2)
            ),
        )
