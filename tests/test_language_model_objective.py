import numpy as np
import pytest

from src.training.language_model_objective import (
    LanguageModelObjective,
)


def test_cross_entropy_should_match_uniform_distribution() -> None:
    objective = LanguageModelObjective()

    logits = np.zeros(
        (
            1,
            2,
            4,
        ),
        dtype=np.float64,
    )

    targets = np.array(
        [
            [0, 1],
        ],
        dtype=np.int64,
    )

    loss = objective.cross_entropy(
        logits,
        targets,
    )

    assert loss == pytest.approx(
        np.log(4.0)
    )


def test_cross_entropy_should_be_lower_for_confident_correct_predictions() -> None:
    objective = LanguageModelObjective()

    logits = np.array(
        [
            [
                [10.0, 0.0],
                [0.0, 10.0],
            ]
        ],
        dtype=np.float64,
    )

    targets = np.array(
        [
            [0, 1],
        ],
        dtype=np.int64,
    )

    loss = objective.cross_entropy(
        logits,
        targets,
    )

    assert loss < 0.001


def test_cross_entropy_should_reject_wrong_logit_rank() -> None:
    objective = LanguageModelObjective()

    with pytest.raises(ValueError):
        objective.cross_entropy(
            np.zeros((2, 3)),
            np.zeros(
                (2, 3),
                dtype=np.int64,
            ),
        )


def test_cross_entropy_should_reject_wrong_target_rank() -> None:
    objective = LanguageModelObjective()

    with pytest.raises(ValueError):
        objective.cross_entropy(
            np.zeros(
                (1, 2, 3)
            ),
            np.zeros(
                (1, 2, 1),
                dtype=np.int64,
            ),
        )


def test_perplexity_should_be_exponential_of_loss() -> None:
    objective = LanguageModelObjective()

    assert objective.perplexity(
        np.log(4.0)
    ) == pytest.approx(4.0)


def test_perplexity_should_reject_non_finite_loss() -> None:
    objective = LanguageModelObjective()

    with pytest.raises(ValueError):
        objective.perplexity(
            float("inf")
        )


def test_perplexity_should_reject_negative_loss() -> None:
    objective = LanguageModelObjective()

    with pytest.raises(ValueError):
        objective.perplexity(-1.0)
