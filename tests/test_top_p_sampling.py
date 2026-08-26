import numpy as np
import pytest

from src.inference.top_p_sampling import (
    TopPSampler,
)


def test_top_p_probabilities_should_sum_to_one() -> None:
    sampler = TopPSampler(
        seed=42
    )

    result = sampler.sample(
        [1.0, 5.0, 3.0, 2.0, 4.0],
        p=0.9,
    )

    assert np.sum(
        result.probabilities
    ) == pytest.approx(1.0)


def test_top_p_should_keep_high_probability_mass() -> None:
    sampler = TopPSampler(
        seed=42
    )

    logits = [
        5.0,
        4.0,
        3.0,
        2.0,
        1.0,
    ]

    result = sampler.sample(
        logits,
        p=0.8,
    )

    probabilities = np.exp(
        np.asarray(logits)
        - np.max(logits)
    )

    probabilities /= np.sum(
        probabilities
    )

    sorted_probabilities = np.sort(
        probabilities
    )[::-1]

    cumulative = np.cumsum(
        sorted_probabilities
    )

    candidate_count = (
        int(
            np.searchsorted(
                cumulative,
                0.8,
                side="left",
            )
        )
        + 1
    )

    assert np.count_nonzero(
        result.probabilities > 0.0
    ) == candidate_count


def test_top_p_should_keep_the_most_probable_token_for_small_p() -> None:
    sampler = TopPSampler(
        seed=42
    )

    result = sampler.sample(
        [10.0, 1.0, 0.0, -1.0],
        p=0.01,
    )

    assert np.count_nonzero(
        result.probabilities > 0.0
    ) == 1

    assert result.probabilities[0] == pytest.approx(
        1.0
    )


def test_top_p_should_allow_full_vocabulary_at_p_one() -> None:
    sampler = TopPSampler(
        seed=42
    )

    result = sampler.sample(
        [1.0, 2.0, 3.0, 4.0],
        p=1.0,
    )

    assert np.count_nonzero(
        result.probabilities > 0.0
    ) == 4


def test_top_p_should_return_valid_token() -> None:
    sampler = TopPSampler(
        seed=42
    )

    result = sampler.sample(
        [1.0, 2.0, 3.0],
        p=0.9,
    )

    assert 0 <= result.token_id < 3


def test_top_p_should_reject_invalid_p() -> None:
    sampler = TopPSampler(
        seed=42
    )

    with pytest.raises(ValueError):
        sampler.sample(
            [1.0, 2.0],
            p=0.0,
        )

    with pytest.raises(ValueError):
        sampler.sample(
            [1.0, 2.0],
            p=1.1,
        )


def test_top_p_should_reject_invalid_temperature() -> None:
    sampler = TopPSampler(
        seed=42
    )

    with pytest.raises(ValueError):
        sampler.sample(
            [1.0, 2.0],
            p=0.9,
            temperature=0.0,
        )
