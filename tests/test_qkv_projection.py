import numpy as np
import pytest

from src.attention.qkv_projection import (
    TrainableQKVProjection,
)


def create_projection() -> TrainableQKVProjection:
    return TrainableQKVProjection(
        input_dimension=4,
        attention_dimension=3,
        seed=42,
    )


def test_qkv_projection_should_return_expected_shapes() -> None:
    projection = create_projection()

    result = projection.forward(
        np.ones(
            (5, 4)
        )
    )

    assert result.queries.shape == (
        5,
        3,
    )

    assert result.keys.shape == (
        5,
        3,
    )

    assert result.values.shape == (
        5,
        3,
    )


def test_qkv_projection_should_use_distinct_parameters() -> None:
    projection = create_projection()

    assert not np.array_equal(
        projection.weights_q,
        projection.weights_k,
    )

    assert not np.array_equal(
        projection.weights_q,
        projection.weights_v,
    )


def test_qkv_backward_should_return_expected_gradient_shapes() -> None:
    projection = create_projection()

    result = projection.forward(
        np.ones(
            (5, 4)
        )
    )

    gradients = projection.backward(
        forward_result=result,
        query_gradient=np.ones_like(
            result.queries
        ),
        key_gradient=np.ones_like(
            result.keys
        ),
        value_gradient=np.ones_like(
            result.values
        ),
    )

    assert gradients.weights_q.shape == (
        4,
        3,
    )

    assert gradients.weights_k.shape == (
        4,
        3,
    )

    assert gradients.weights_v.shape == (
        4,
        3,
    )

    assert gradients.input.shape == (
        5,
        4,
    )


def test_qkv_backward_should_accumulate_all_input_paths() -> None:
    projection = create_projection()

    inputs = np.asarray(
        [
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 1.0, 0.0, 3.0],
        ]
    )

    result = projection.forward(
        inputs
    )

    q_gradient = np.ones_like(
        result.queries
    )

    k_gradient = np.zeros_like(
        result.keys
    )

    v_gradient = np.zeros_like(
        result.values
    )

    gradients = projection.backward(
        forward_result=result,
        query_gradient=q_gradient,
        key_gradient=k_gradient,
        value_gradient=v_gradient,
    )

    expected = (
        q_gradient
        @ projection.weights_q.T
    )

    np.testing.assert_allclose(
        gradients.input,
        expected,
    )


def test_qkv_training_step_should_change_weights() -> None:
    projection = create_projection()

    before_q = projection.weights_q
    before_k = projection.weights_k
    before_v = projection.weights_v

    result = projection.forward(
        np.ones(
            (3, 4)
        )
    )

    gradients = projection.backward(
        forward_result=result,
        query_gradient=np.ones_like(
            result.queries
        ),
        key_gradient=np.ones_like(
            result.keys
        ),
        value_gradient=np.ones_like(
            result.values
        ),
    )

    projection.apply_gradients(
        gradients,
        learning_rate=0.01,
    )

    assert not np.array_equal(
        before_q,
        projection.weights_q,
    )

    assert not np.array_equal(
        before_k,
        projection.weights_k,
    )

    assert not np.array_equal(
        before_v,
        projection.weights_v,
    )


def test_qkv_projection_should_reject_wrong_input_dimension() -> None:
    projection = create_projection()

    with pytest.raises(ValueError):
        projection.forward(
            np.ones(
                (3, 5)
            )
        )


def test_qkv_projection_should_reject_invalid_learning_rate() -> None:
    projection = create_projection()

    result = projection.forward(
        np.ones(
            (3, 4)
        )
    )

    gradients = projection.backward(
        forward_result=result,
        query_gradient=np.ones_like(
            result.queries
        ),
        key_gradient=np.ones_like(
            result.keys
        ),
        value_gradient=np.ones_like(
            result.values
        ),
    )

    with pytest.raises(ValueError):
        projection.apply_gradients(
            gradients,
            learning_rate=0.0,
        )

from src.attention.qkv_projection import (
    TrainableQKVProjection,
)


def test_qkv_projection_q_gradient_should_match_numerical_gradient() -> None:
    projection = TrainableQKVProjection(
        input_dimension=3,
        attention_dimension=2,
        seed=42,
    )

    inputs = np.asarray(
        [
            [0.2, -0.5, 0.7],
            [1.1, 0.3, -0.4],
        ],
        dtype=np.float64,
    )

    forward = projection.forward(
        inputs
    )

    query_gradient = np.asarray(
        [
            [0.4, -0.2],
            [0.7, 0.5],
        ],
        dtype=np.float64,
    )

    key_gradient = np.zeros_like(
        forward.keys
    )

    value_gradient = np.zeros_like(
        forward.values
    )

    analytical = projection.backward(
        forward_result=forward,
        query_gradient=query_gradient,
        key_gradient=key_gradient,
        value_gradient=value_gradient,
    )

    row = 1
    column = 0

    epsilon = 1e-6

    original = projection.weights_q[
        row,
        column,
    ]

    projection._weights_q[
        row,
        column
    ] = original + epsilon

    plus = projection.forward(
        inputs
    ).queries

    projection._weights_q[
        row,
        column
    ] = original - epsilon

    minus = projection.forward(
        inputs
    ).queries

    projection._weights_q[
        row,
        column
    ] = original

    numerical = np.sum(
        (
            plus
            - minus
        )
        * query_gradient
    ) / (
        2.0
        * epsilon
    )

    np.testing.assert_allclose(
        analytical.weights_q[
            row,
            column,
        ],
        numerical,
        rtol=1e-5,
        atol=1e-7,
    )
