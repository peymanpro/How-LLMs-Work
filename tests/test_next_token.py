import numpy as np
import pytest

from src.inference.next_token import (
    NextTokenPredictor,
)


def test_predictor_should_select_highest_probability_token() -> None:
    predictor = NextTokenPredictor()

    result = predictor.predict(
        [
            [1.0, 2.0, 3.0],
            [0.0, 5.0, 1.0],
        ]
    )

    assert result.token_id == 1


def test_predictor_probabilities_should_sum_to_one() -> None:
    predictor = NextTokenPredictor()

    result = predictor.predict(
        [
            [1.0, 2.0, 3.0],
        ]
    )

    assert np.sum(
        result.probabilities
    ) == pytest.approx(1.0)


def test_predictor_should_return_one_probability_per_vocabulary_token() -> None:
    predictor = NextTokenPredictor()

    result = predictor.predict(
        [
            [1.0, 2.0, 3.0, 4.0],
        ]
    )

    assert result.probabilities.shape == (
        4,
    )


def test_predictor_should_use_only_last_position() -> None:
    predictor = NextTokenPredictor()

    result = predictor.predict(
        [
            [100.0, 0.0, 0.0],
            [0.0, 100.0, 0.0],
        ]
    )

    assert result.token_id == 1


def test_predictor_should_reject_empty_sequence() -> None:
    predictor = NextTokenPredictor()

    with pytest.raises(ValueError):
        predictor.predict([])


def test_predictor_should_reject_wrong_rank() -> None:
    predictor = NextTokenPredictor()

    with pytest.raises(ValueError):
        predictor.predict(
            [
                1.0,
                2.0,
                3.0,
            ]
        )
