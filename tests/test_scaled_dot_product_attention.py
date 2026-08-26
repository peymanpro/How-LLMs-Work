import numpy as np
import pytest

from src.attention.scaled_dot_product import (
    ScaledDotProductAttention,
)


def test_attention_should_return_expected_shapes() -> None:
    attention = ScaledDotProductAttention()

    result = attention.forward(
        np.ones((3, 4)),
        np.ones((3, 4)),
        np.ones((3, 5)),
    )

    assert result.scores.shape == (3, 3)
    assert result.weights.shape == (3, 3)
    assert result.output.shape == (3, 5)


def test_attention_weights_should_sum_to_one() -> None:
    attention = ScaledDotProductAttention()

    result = attention.forward(
        np.asarray(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        ),
        np.asarray(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        ),
        np.asarray(
            [
                [2.0, 3.0],
                [5.0, 7.0],
            ]
        ),
    )

    np.testing.assert_allclose(
        result.weights.sum(axis=1),
        np.ones(2),
    )


def test_causal_attention_should_hide_future_tokens() -> None:
    attention = ScaledDotProductAttention(
        causal=True
    )

    result = attention.forward(
        np.asarray(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ]
        ),
        np.asarray(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ]
        ),
        np.asarray(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [10.0, 10.0],
            ]
        ),
    )

    assert result.weights[0, 1] == 0.0
    assert result.weights[0, 2] == 0.0
    assert result.weights[1, 2] == 0.0


def test_first_causal_position_can_only_see_itself() -> None:
    attention = ScaledDotProductAttention(
        causal=True
    )

    result = attention.forward(
        np.asarray(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        ),
        np.asarray(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        ),
        np.asarray(
            [
                [3.0, 4.0],
                [100.0, 200.0],
            ]
        ),
    )

    np.testing.assert_allclose(
        result.output[0],
        [3.0, 4.0],
    )


def test_query_and_key_dimensions_must_match() -> None:
    attention = ScaledDotProductAttention()

    with pytest.raises(ValueError):
        attention.forward(
            np.ones((2, 3)),
            np.ones((2, 4)),
            np.ones((2, 5)),
        )


def test_key_and_value_lengths_must_match() -> None:
    attention = ScaledDotProductAttention()

    with pytest.raises(ValueError):
        attention.forward(
            np.ones((2, 3)),
            np.ones((3, 3)),
            np.ones((2, 4)),
        )


def test_causal_attention_requires_equal_sequence_lengths() -> None:
    attention = ScaledDotProductAttention(
        causal=True
    )

    with pytest.raises(ValueError):
        attention.forward(
            np.ones((1, 3)),
            np.ones((3, 3)),
            np.ones((3, 4)),
        )
