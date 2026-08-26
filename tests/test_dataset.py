import pytest

from src.training.dataset import (
    BatchBuilder,
    LanguageModelDataset,
)


def test_dataset_should_create_shifted_context_windows() -> None:
    dataset = LanguageModelDataset(
        token_ids=[
            0,
            1,
            2,
            3,
            4,
            5,
        ],
        context_size=3,
    )

    assert dataset.size == 3

    assert dataset.get(0).inputs == (
        0,
        1,
        2,
    )

    assert dataset.get(0).targets == (
        1,
        2,
        3,
    )

    assert dataset.get(2).inputs == (
        2,
        3,
        4,
    )

    assert dataset.get(2).targets == (
        3,
        4,
        5,
    )


def test_dataset_should_reject_too_short_stream() -> None:
    with pytest.raises(ValueError):
        LanguageModelDataset(
            token_ids=[1, 2, 3],
            context_size=3,
        )


def test_dataset_should_reject_invalid_context_size() -> None:
    with pytest.raises(ValueError):
        LanguageModelDataset(
            token_ids=[1, 2, 3],
            context_size=1,
        )


def test_dataset_should_reject_empty_stream() -> None:
    with pytest.raises(ValueError):
        LanguageModelDataset(
            token_ids=[],
            context_size=3,
        )


def test_dataset_should_reject_invalid_index() -> None:
    dataset = LanguageModelDataset(
        token_ids=[
            0,
            1,
            2,
            3,
        ],
        context_size=2,
    )

    with pytest.raises(IndexError):
        dataset.get(10)


def test_batch_builder_should_group_examples() -> None:
    dataset = LanguageModelDataset(
        token_ids=[
            0,
            1,
            2,
            3,
            4,
            5,
        ],
        context_size=3,
    )

    batches = BatchBuilder().create_batches(
        dataset,
        batch_size=2,
    )

    assert len(batches) == 2

    assert batches[0].inputs == (
        (0, 1, 2),
        (1, 2, 3),
    )

    assert batches[0].targets == (
        (1, 2, 3),
        (2, 3, 4),
    )


def test_batch_builder_should_keep_final_smaller_batch() -> None:
    dataset = LanguageModelDataset(
        token_ids=[
            0,
            1,
            2,
            3,
            4,
            5,
            6,
        ],
        context_size=3,
    )

    batches = BatchBuilder().create_batches(
        dataset,
        batch_size=2,
    )

    assert len(batches) == 2

    assert len(
        batches[-1].inputs
    ) == 2


def test_batch_builder_should_reject_non_positive_batch_size() -> None:
    dataset = LanguageModelDataset(
        token_ids=[
            0,
            1,
            2,
        ],
        context_size=2,
    )

    with pytest.raises(ValueError):
        BatchBuilder().create_batches(
            dataset,
            batch_size=0,
        )
