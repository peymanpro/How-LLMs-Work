import pytest

from src.training.causal_examples import (
    CausalExampleBuilder,
)


def test_causal_example_builder_should_shift_sequence_by_one() -> None:
    result = CausalExampleBuilder().build(
        [1, 2, 3, 4]
    )

    assert result.inputs == (
        1,
        2,
        3,
    )

    assert result.targets == (
        2,
        3,
        4,
    )


def test_causal_example_should_have_equal_input_and_target_length() -> None:
    result = CausalExampleBuilder().build(
        [5, 7, 9]
    )

    assert len(result.inputs) == len(
        result.targets
    )


def test_causal_example_builder_should_reject_one_token_sequence() -> None:
    with pytest.raises(ValueError):
        CausalExampleBuilder().build(
            [1]
        )


def test_causal_example_builder_should_reject_empty_sequence() -> None:
    with pytest.raises(ValueError):
        CausalExampleBuilder().build(
            []
        )
