import numpy as np
import pytest

from src.inference.sampling import (
    TemperatureSampler,
)


def test_sampler_probabilities_should_sum_to_one() -> None:
    sampler = TemperatureSampler(
        seed=42
    )

    result = sampler.sample(
        [1.0, 2.0, 3.0],
        temperature=1.0,
    )

    assert np.sum(
        result.probabilities
    ) == pytest.approx(1.0)


def test_sampler_should_return_valid_token() -> None:
    sampler = TemperatureSampler(
        seed=42
    )

    result = sampler.sample(
        [1.0, 2.0, 3.0],
        temperature=1.0,
    )

    assert 0 <= result.token_id < 3


def test_lower_temperature_should_make_distribution_sharper() -> None:
    sampler = TemperatureSampler(
        seed=42
    )

    low = sampler.sample(
        [1.0, 2.0, 3.0],
        temperature=0.2,
    )

    high = sampler.sample(
        [1.0, 2.0, 3.0],
        temperature=2.0,
    )

    assert low.probabilities.max() > (
        high.probabilities.max()
    )


def test_temperature_should_not_change_probability_order() -> None:
    sampler = TemperatureSampler(
        seed=42
    )

    result = sampler.sample(
        [0.5, 2.0, 1.0],
        temperature=0.5,
    )

    assert np.argsort(
        result.probabilities
    ).tolist() == [0, 2, 1]


def test_sampler_should_reject_non_positive_temperature() -> None:
    sampler = TemperatureSampler(
        seed=42
    )

    with pytest.raises(ValueError):
        sampler.sample(
            [1.0, 2.0],
            temperature=0.0,
        )


def test_sampler_should_reject_empty_logits() -> None:
    sampler = TemperatureSampler(
        seed=42
    )

    with pytest.raises(ValueError):
        sampler.sample(
            [],
            temperature=1.0,
        )
