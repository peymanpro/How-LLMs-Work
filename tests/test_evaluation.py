import numpy as np
import pytest

from src.training.evaluation import (
    LanguageModelEvaluator,
)


def test_evaluator_should_return_loss_and_perplexity() -> None:
    evaluator = LanguageModelEvaluator()

    logits = np.zeros(
        (
            2,
            3,
            4,
        ),
        dtype=np.float64,
    )

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

    assert result.loss == pytest.approx(
        np.log(4.0)
    )

    assert result.perplexity == pytest.approx(
        4.0
    )
