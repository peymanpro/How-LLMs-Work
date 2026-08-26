import numpy as np

from src.inference.generator import (
    TextGenerator,
)
from src.inference.sampling_strategy import (
    GreedySamplingStrategy,
    TemperatureSamplingStrategy,
    TopKSamplingStrategy,
    TopPSamplingStrategy,
)


class FakeInference:
    def __init__(
        self,
        logits: list[list[float]],
    ) -> None:
        self._logits = logits

    def next_logits(
        self,
        token_ids: list[int],
    ) -> list[float]:
        del token_ids

        return self._logits[0]


def test_generator_should_use_greedy_strategy() -> None:
    inference = FakeInference(
        [[1.0, 5.0, 2.0]]
    )

    generator = TextGenerator(
        inference=inference,
        context_size=1,
        strategy=GreedySamplingStrategy(),
    )

    result = generator.generate(
        prompt=[0],
        max_new_tokens=1,
    )

    assert result.token_ids == (
        0,
        1,
    )


def test_generator_should_use_temperature_strategy() -> None:
    inference = FakeInference(
        [[1.0, 2.0, 3.0]]
    )

    generator = TextGenerator(
        inference=inference,
        context_size=1,
        strategy=TemperatureSamplingStrategy(
            temperature=1.0,
            seed=42,
        ),
    )

    result = generator.generate(
        prompt=[0],
        max_new_tokens=3,
    )

    assert len(result.token_ids) == 4


def test_generator_should_use_top_k_strategy() -> None:
    inference = FakeInference(
        [[1.0, 5.0, 3.0, 0.0]]
    )

    generator = TextGenerator(
        inference=inference,
        context_size=1,
        strategy=TopKSamplingStrategy(
            k=2,
            temperature=1.0,
            seed=42,
        ),
    )

    result = generator.generate(
        prompt=[0],
        max_new_tokens=10,
    )

    assert all(
        token_id in {0, 1, 2, 3}
        for token_id in result.token_ids
    )


def test_generator_should_use_top_p_strategy() -> None:
    inference = FakeInference(
        [[1.0, 5.0, 3.0, 0.0]]
    )

    generator = TextGenerator(
        inference=inference,
        context_size=1,
        strategy=TopPSamplingStrategy(
            p=0.9,
            temperature=1.0,
            seed=42,
        ),
    )

    result = generator.generate(
        prompt=[0],
        max_new_tokens=10,
    )

    assert len(result.token_ids) == 11


def test_generator_should_preserve_prompt() -> None:
    inference = FakeInference(
        [[1.0, 5.0]]
    )

    generator = TextGenerator(
        inference=inference,
        context_size=2,
        strategy=GreedySamplingStrategy(),
    )

    result = generator.generate(
        prompt=[0, 1],
        max_new_tokens=1,
    )

    assert result.token_ids[:2] == (
        0,
        1,
    )


def test_greedy_and_sampling_strategy_can_share_same_interface() -> None:
    logits = np.asarray(
        [1.0, 4.0, 2.0],
        dtype=np.float64,
    )

    greedy = GreedySamplingStrategy()

    sampled = TemperatureSamplingStrategy(
        temperature=1.0,
        seed=42,
    )

    assert greedy.sample(
        logits
    ) == 1

    assert 0 <= sampled.sample(
        logits
    ) < 3
