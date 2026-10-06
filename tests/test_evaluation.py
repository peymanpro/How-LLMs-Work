import numpy as np
import pytest

from src.training.evaluation import (
    LanguageModelEvaluator,
)


def test_evaluator_should_return_loss_perplexity_and_accuracy() -> None:
    evaluator = LanguageModelEvaluator()

    logits = np.zeros(
        (
            2,
            3,
            4,
        ),
        dtype=np.float64,
    )

    logits[:, :, 0] = 1.0

    targets = np.array(
        [
            [0, 1, 2],
            [1, 2, 3],
        ],
        dtype=np.int64,
    )

    result = evaluator.evaluate(
        logits,
        targets,
    )

    log_normalizer = np.log(
        np.exp(1.0) + 3.0
    )
    expected_loss = (
        log_normalizer - (1.0 / 6.0)
    )

    assert result.loss == pytest.approx(
        expected_loss
    )

    assert result.perplexity == pytest.approx(
        np.exp(expected_loss)
    )

    assert result.token_accuracy == pytest.approx(
        1.0 / 6.0
    )
