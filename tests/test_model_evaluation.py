import numpy as np
import pytest

from src.llm.transition_model import TransitionTableLanguageModel
from src.training.dataset import BatchBuilder, LanguageModelDataset
from src.training.model_evaluation import BatchLanguageModelEvaluator


def create_model() -> TransitionTableLanguageModel:
    return TransitionTableLanguageModel(
        vocabulary_size=4,
        context_size=3,
        transitions={
            0: 1,
            1: 2,
            2: 3,
        },
    )


def test_transition_model_should_return_expected_logit_shape() -> None:
    model = create_model()

    logits = model.logits([0, 1, 2])

    assert logits.shape == (3, 4)


def test_transition_model_should_assign_high_logit_to_target_transition() -> None:
    model = create_model()

    logits = model.logits([0, 1, 2])

    np.testing.assert_array_equal(
        np.argmax(logits, axis=1),
        np.array([1, 2, 3]),
    )


def test_model_evaluator_should_calculate_low_loss_for_correct_transitions() -> None:
    model = create_model()

    dataset = LanguageModelDataset(
        token_ids=[0, 1, 2, 3],
        context_size=3,
    )

    batch = BatchBuilder().create_batches(
        dataset,
        batch_size=1,
    )[0]

    result = BatchLanguageModelEvaluator().evaluate_batch(
        model,
        batch,
    )

    assert result.examples == 1
    assert result.loss < 0.001
    assert result.perplexity < 1.01


def test_model_evaluator_should_reject_wrong_context_length() -> None:
    model = create_model()

    with pytest.raises(ValueError):
        model.logits([0, 1])


def test_transition_model_should_reject_invalid_token() -> None:
    model = create_model()

    with pytest.raises(IndexError):
        model.logits([0, 1, 9])
