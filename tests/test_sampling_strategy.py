import numpy as np
import pytest

from src.inference.sampling_strategy import (
    GreedySamplingStrategy,
    TemperatureSamplingStrategy,
    TopKSamplingStrategy,
    TopPSamplingStrategy,
)


def test_greedy_strategy_should_select_maximum_logit() -> None:
    strategy = GreedySamplingStrategy()

    assert strategy.sample(
        np.asarray(
            [1.0, 5.0, 2.0]
        )
    ) == 1


def test_temperature_strategy_should_return_valid_token() -> None:
    strategy = TemperatureSamplingStrategy(
        temperature=1.0,
        seed=42,
    )

    token_id = strategy.sample(
        np.asarray(
            [1.0, 2.0, 3.0]
        )
    )

    assert 0 <= token_id < 3


def test_top_k_strategy_should_return_only_top_k_candidates() -> None:
    strategy = TopKSamplingStrategy(
        k=2,
        temperature=1.0,
        seed=42,
    )

    for _ in range(100):
        token_id = strategy.sample(
            np.asarray(
                [1.0, 5.0, 3.0, 2.0]
            )
        )

        assert token_id in {
            1,
            2,
        }


def test_top_p_strategy_should_return_valid_token() -> None:
    strategy = TopPSamplingStrategy(
        p=0.9,
        temperature=1.0,
        seed=42,
    )

    token_id = strategy.sample(
        np.asarray(
            [1.0, 2.0, 3.0]
        )
    )

    assert 0 <= token_id < 3


@pytest.mark.parametrize(
    "strategy",
    [
        TemperatureSamplingStrategy(
            temperature=1.0,
            seed=42,
        ),
        TopKSamplingStrategy(
            k=2,
            temperature=1.0,
            seed=42,
        ),
        TopPSamplingStrategy(
            p=0.9,
            temperature=1.0,
            seed=42,
        ),
    ],
)
def test_stochastic_strategies_should_return_python_int(
    strategy,
) -> None:
    token_id = strategy.sample(
        np.asarray(
            [1.0, 2.0, 3.0]
        )
    )

    assert isinstance(
        token_id,
        int,
    )
