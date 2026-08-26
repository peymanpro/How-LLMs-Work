import numpy as np
import pytest

from src.attention.multi_head import (
    TrainableMultiHeadAttention,
)


def create_attention() -> TrainableMultiHeadAttention:
    return TrainableMultiHeadAttention(
        model_dimension=8,
        number_of_heads=2,
        seed=42,
        causal=True,
    )


def test_multi_head_attention_should_return_expected_shapes() -> None:
    attention = create_attention()

    result = attention.forward(
        np.ones(
            (4, 8)
        )
    )

    assert len(result.heads) == 2

    assert result.heads[0].output.shape == (
        4,
        4,
    )

    assert result.heads[1].output.shape == (
        4,
        4,
    )

    assert result.concatenated.shape == (
        4,
        8,
    )

    assert result.output.shape == (
        4,
        8,
    )


def test_multi_head_attention_should_split_model_dimension_across_heads() -> None:
    attention = create_attention()

    assert attention.head_dimension == 4
    assert attention.number_of_heads == 2


def test_multi_head_attention_should_use_distinct_head_attention() -> None:
    attention = create_attention()

    inputs = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        ]
    )

    result = attention.forward(
        inputs
    )

    assert len(
        result.heads
    ) == 2

    assert result.heads[0].weights.shape == (
        3,
        3,
    )

    assert result.heads[1].weights.shape == (
        3,
        3,
    )


def test_multi_head_attention_should_apply_causal_mask() -> None:
    attention = create_attention()

    result = attention.forward(
        np.eye(
            4,
            8,
        )
    )

    for head in result.heads:
        assert head.weights[0, 1] == 0.0
        assert head.weights[0, 2] == 0.0
        assert head.weights[0, 3] == 0.0

        assert head.weights[1, 2] == 0.0
        assert head.weights[1, 3] == 0.0

        assert head.weights[2, 3] == 0.0


def test_multi_head_attention_should_produce_finite_output() -> None:
    attention = create_attention()

    result = attention.forward(
        np.random.default_rng(
            42
        ).normal(
            size=(5, 8)
        )
    )

    assert np.isfinite(
        result.output
    ).all()


def test_multi_head_attention_should_reject_non_divisible_dimensions() -> None:
    with pytest.raises(ValueError):
        TrainableMultiHeadAttention(
            model_dimension=7,
            number_of_heads=2,
        )


def test_multi_head_attention_should_reject_invalid_head_count() -> None:
    with pytest.raises(ValueError):
        TrainableMultiHeadAttention(
            model_dimension=8,
            number_of_heads=0,
        )


def test_multi_head_attention_should_reject_wrong_input_dimension() -> None:
    attention = create_attention()

    with pytest.raises(ValueError):
        attention.forward(
            np.ones(
                (4, 7)
            )
        )
