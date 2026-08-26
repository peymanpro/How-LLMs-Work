import numpy as np

from src.llm.positional_language_model import (
    PositionalContextLanguageModel,
)
from src.training.positional_language_model_training import (
    PositionalLanguageModelTrainingStep,
)


def create_model() -> PositionalContextLanguageModel:
    return PositionalContextLanguageModel(
        vocabulary_size=5,
        context_size=3,
        model_dimension=8,
        seed=42,
    )


def test_positional_model_should_return_expected_shape() -> None:
    model = create_model()

    result = model.forward(
        [0, 1, 2]
    )

    assert result.logits.shape == (
        3,
        5,
    )

    assert result.hidden.shape == (
        3,
        8,
    )


def test_positional_model_should_be_order_sensitive() -> None:
    model = create_model()

    first = model.forward(
        [0, 1, 2]
    )

    reordered = model.forward(
        [2, 1, 0]
    )

    assert not np.allclose(
        first.hidden,
        reordered.hidden,
    )

    assert not np.allclose(
        first.logits,
        reordered.logits,
    )


def test_positional_model_should_train() -> None:
    model = create_model()

    step = PositionalLanguageModelTrainingStep(
        learning_rate=0.05,
    )

    first = step.run(
        model=model,
        token_ids=[0, 1, 2],
        targets=[1, 2, 3],
    )

    result = first

    for _ in range(100):
        result = step.run(
            model=model,
            token_ids=[0, 1, 2],
            targets=[1, 2, 3],
        )

    assert result.loss < first.loss
    assert result.perplexity < first.perplexity
def test_positional_model_position_gradient_should_match_numerical_gradient() -> None:
    model = create_model()

    token_ids = [0, 1, 2]

    forward = model.forward(
        token_ids
    )

    targets = np.array(
        [1, 2, 3],
        dtype=np.int64,
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

    output_gradient = probabilities.copy()

    output_gradient[
        np.arange(3),
        targets,
    ] -= 1.0

    output_gradient /= 3.0

    analytical = model.backward(
        token_ids=token_ids,
        forward_result=forward,
        output_gradient=output_gradient,
    )

    epsilon = 1e-6

    position = 1
    dimension = 2

    original = model._position_embeddings[
        position,
        dimension,
    ]

    model._position_embeddings[
        position,
        dimension,
    ] = original + epsilon

    plus = model.forward(
        token_ids
    ).logits

    model._position_embeddings[
        position,
        dimension,
    ] = original - epsilon

    minus = model.forward(
        token_ids
    ).logits

    model._position_embeddings[
        position,
        dimension,
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

        target_logits = shifted_logits[
            np.arange(3),
            targets,
        ]

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
        analytical.position_embeddings[
            position,
            dimension,
        ],
        numerical,
        rtol=1e-4,
        atol=1e-5,
    )
