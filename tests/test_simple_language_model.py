import numpy as np
import pytest

from src.llm.simple_language_model import (
    SimpleContextLanguageModel,
)
from src.training.language_model_training import (
    LanguageModelTrainingStep,
)


def create_model() -> SimpleContextLanguageModel:
    return SimpleContextLanguageModel(
        vocabulary_size=5,
        context_size=3,
        model_dimension=8,
        seed=42,
    )


def test_model_should_return_expected_logit_shape() -> None:
    model = create_model()

    result = model.forward(
        [0, 1, 2]
    )

    assert result.logits.shape == (
        3,
        5,
    )

    assert result.hidden.shape == (
        8,
    )


def test_model_should_implement_language_model_interface() -> None:
    model = create_model()

    logits = model.logits(
        [0, 1, 2]
    )

    assert logits.shape == (
        3,
        5,
    )


def test_model_should_reject_wrong_context_length() -> None:
    model = create_model()

    with pytest.raises(ValueError):
        model.forward(
            [0, 1]
        )


def test_model_should_reject_invalid_token() -> None:
    model = create_model()

    with pytest.raises(IndexError):
        model.forward(
            [0, 1, 9]
        )


def test_training_step_should_update_model() -> None:
    model = create_model()

    before = model.forward(
        [0, 1, 2]
    ).logits.copy()

    LanguageModelTrainingStep(
        learning_rate=0.05,
    ).run(
        model=model,
        token_ids=[0, 1, 2],
        targets=[1, 2, 3],
    )

    after = model.forward(
        [0, 1, 2]
    ).logits

    assert not np.array_equal(
        before,
        after,
    )


def test_training_should_reduce_loss() -> None:
    model = create_model()

    step = LanguageModelTrainingStep(
        learning_rate=0.05,
    )

    first = step.run(
        model=model,
        token_ids=[0, 1, 2],
        targets=[1, 2, 3],
    )

    last = first

    for _ in range(100):
        last = step.run(
            model=model,
            token_ids=[0, 1, 2],
            targets=[1, 2, 3],
        )

    assert last.loss < first.loss


def test_training_should_reduce_perplexity() -> None:
    model = create_model()

    step = LanguageModelTrainingStep(
        learning_rate=0.05,
    )

    first = step.run(
        model=model,
        token_ids=[0, 1, 2],
        targets=[1, 2, 3],
    )

    last = first

    for _ in range(100):
        last = step.run(
            model=model,
            token_ids=[0, 1, 2],
            targets=[1, 2, 3],
        )

    assert last.perplexity < first.perplexity
def test_model_input_embedding_gradient_should_match_numerical_gradient() -> None:
    model = create_model()

    token_ids = [0, 1, 2]
    targets = np.array(
        [1, 2, 3],
        dtype=np.int64,
    )

    forward = model.forward(
        token_ids
    )

    shifted = (
        forward.logits
        - np.max(
            forward.logits,
            axis=1,
            keepdims=True,
        )
    )

    probabilities = np.exp(
        shifted
    )

    probabilities /= np.sum(
        probabilities,
        axis=1,
        keepdims=True,
    )

    gradient = probabilities.copy()

    gradient[
        np.arange(3),
        targets,
    ] -= 1.0

    gradient /= 3.0

    analytical = model.backward(
        token_ids=token_ids,
        forward_result=forward,
        output_gradient=gradient,
    )

    epsilon = 1e-6

    row = 1
    column = 2

    original = model._embeddings[
        row,
        column,
    ]

    model._embeddings[
        row,
        column,
    ] = original + epsilon

    plus = model.forward(
        token_ids
    ).logits

    model._embeddings[
        row,
        column,
    ] = original - epsilon

    minus = model.forward(
        token_ids
    ).logits

    model._embeddings[
        row,
        column,
    ] = original

    def loss(
        logits: np.ndarray,
    ) -> float:
        shifted_logits = (
            logits
            - np.max(
                logits,
                axis=1,
                keepdims=True,
            )
        )

        log_sum_exp = np.log(
            np.sum(
                np.exp(
                    shifted_logits
                ),
                axis=1,
            )
        )

        target_logits = (
            shifted_logits[
                np.arange(3),
                targets,
            ]
        )

        return float(
            np.mean(
                -target_logits
                + log_sum_exp
            )
        )

    numerical = (
        loss(plus)
        - loss(minus)
    ) / (
        2.0 * epsilon
    )

    np.testing.assert_allclose(
        analytical.embeddings[
            row,
            column,
        ],
        numerical,
        rtol=1e-4,
        atol=1e-5,
    )

