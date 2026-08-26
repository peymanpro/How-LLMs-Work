from dataclasses import dataclass

import pytest

from src.inference.generator import (
    GreedyTextGenerator,
)


@dataclass(frozen=True)
class FakePrediction:
    token_id: int


@dataclass(frozen=True)
class FakeStep:
    prediction: FakePrediction


class FakeInference:
    def __init__(
        self,
        predictions: list[int],
    ) -> None:
        self._predictions = predictions
        self._index = 0
        self.contexts: list[list[int]] = []

    def predict_next(
        self,
        token_ids: list[int],
    ) -> FakeStep:
        self.contexts.append(
            list(token_ids)
        )

        token_id = self._predictions[
            self._index
        ]

        self._index += 1

        return FakeStep(
            prediction=FakePrediction(
                token_id=token_id
            )
        )


def test_generator_should_append_predicted_tokens() -> None:
    inference = FakeInference(
        [3, 4, 2]
    )

    generator = GreedyTextGenerator(
        inference=inference,
        context_size=4,
    )

    result = generator.generate(
        prompt=[0, 1, 2, 3],
        max_new_tokens=3,
    )

    assert result.token_ids == (
        0,
        1,
        2,
        3,
        3,
        4,
        2,
    )


def test_generator_should_keep_only_context_window() -> None:
    inference = FakeInference(
        [3, 4, 2]
    )

    generator = GreedyTextGenerator(
        inference=inference,
        context_size=3,
    )

    generator.generate(
        prompt=[0, 1, 2],
        max_new_tokens=3,
    )

    assert inference.contexts == [
        [0, 1, 2],
        [1, 2, 3],
        [2, 3, 4],
    ]


def test_generator_should_stop_on_eos() -> None:
    inference = FakeInference(
        [4, 2, 3]
    )

    generator = GreedyTextGenerator(
        inference=inference,
        context_size=4,
    )

    result = generator.generate(
        prompt=[0, 1, 2, 3],
        max_new_tokens=3,
        eos_token_id=4,
    )

    assert result.token_ids == (
        0,
        1,
        2,
        3,
        4,
    )


def test_generator_should_reject_empty_prompt() -> None:
    inference = FakeInference(
        [1]
    )

    generator = GreedyTextGenerator(
        inference=inference,
        context_size=4,
    )

    with pytest.raises(ValueError):
        generator.generate(
            prompt=[],
            max_new_tokens=1,
        )


def test_generator_should_reject_negative_token_count() -> None:
    inference = FakeInference(
        [1]
    )

    generator = GreedyTextGenerator(
        inference=inference,
        context_size=4,
    )

    with pytest.raises(ValueError):
        generator.generate(
            prompt=[0],
            max_new_tokens=-1,
        )


def test_generator_should_allow_zero_new_tokens() -> None:
    inference = FakeInference(
        [1]
    )

    generator = GreedyTextGenerator(
        inference=inference,
        context_size=4,
    )

    result = generator.generate(
        prompt=[0, 1],
        max_new_tokens=0,
    )

    assert result.token_ids == (
        0,
        1,
    )
def test_generator_should_reject_prompt_shorter_than_context_size() -> None:
    inference = FakeInference(
        [1]
    )

    generator = GreedyTextGenerator(
        inference=inference,
        context_size=4,
    )

    with pytest.raises(ValueError):
        generator.generate(
            prompt=[0, 1],
            max_new_tokens=1,
        )


