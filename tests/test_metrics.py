import numpy as np
import pytest

from src.training.metrics import (
    LanguageModelMetrics,
)


def test_token_accuracy_should_return_fraction_of_correct_predictions() -> None:
    logits = np.asarray(
        [
            [
                [3.0, 1.0, 0.0],
                [0.0, 2.0, 1.0],
                [0.0, 1.0, 2.0],
                [3.0, 0.0, 1.0],
            ]
        ],
        dtype=np.float64,
    )

    targets = np.asarray(
        [
            [0, 1, 1, 2],
        ],
        dtype=np.int64,
    )

    accuracy = LanguageModelMetrics.token_accuracy(
        logits,
        targets,
    )

    assert accuracy == pytest.approx(0.75)


def test_token_accuracy_should_reject_invalid_target_id() -> None:
    logits = np.zeros(
        (
            1,
            2,
            3,
        ),
        dtype=np.float64,
    )

    targets = np.asarray(
        [
            [0, 3],
        ],
        dtype=np.int64,
    )

    with pytest.raises(ValueError):
        LanguageModelMetrics.token_accuracy(
            logits,
            targets,
        )
