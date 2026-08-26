import numpy as np
import pytest

from src.inference.top_k_sampling import (
    TopKSampler,
)


def test_top_k_should_keep_exactly_k_candidates() -> None:
    sampler = TopKSampler(
        seed=42
    )

    result = sampler.sample(
        [1.0, 5.0, 3.0, 2.0, 4.0],
        k=2,
    )

    non_zero = np.count_nonzero(
        result.probabilities > 0.0
    )

    assert non_zero == 2


def test_top_k_should_keep_highest_logits() -> None:
    sampler = TopKSampler(
        seed=42
    )

    result = sampler.sample(
        [1.0, 5.0, 3.0, 2.0, 4.0],
        k=2,
    )

    assert result.probabilities[1] > 0.0
    assert result.probabilities[4] > 0.0

    assert result.probabilities[0] == 0.0
    assert result.probabilities[2] == 0.0
    assert result.probabilities[3] == 0.0


def test_top_k_probabilities_should_sum_to_one() -> None:
    sampler = TopKSampler(
        seed=42
    )

    result = sampler.sample(
        [1.0, 5.0, 3.0, 2.0, 4.0],
        k=3,
        temperature=1.5,
    )

    assert np.sum(
        result.probabilities
    ) == pytest.approx(1.0)


def test_top_k_should_return_valid_token() -> None:
    sampler = TopKSampler(
        seed=42
    )

    result = sampler.sample(
        [1.0, 5.0, 3.0, 2.0, 4.0],
        k=2,
    )

    assert result.token_id in {
        1,
        4,
    }


def test_top_k_should_reject_non_positive_k() -> None:
    sampler = TopKSampler(
        seed=42
    )

    with pytest.raises(ValueError):
        sampler.sample(
            [1.0, 2.0],
            k=0,
        )


def test_top_k_should_reject_k_larger_than_vocabulary() -> None:
    sampler = TopKSampler(
        seed=42
    )

    with pytest.raises(ValueError):
        sampler.sample(
            [1.0, 2.0],
            k=3,
        )


def test_top_k_should_reject_non_positive_temperature() -> None:
    sampler = TopKSampler(
        seed=42
    )

    with pytest.raises(ValueError):
        sampler.sample(
            [1.0, 2.0],
            k=1,
            temperature=0.0,
        )
