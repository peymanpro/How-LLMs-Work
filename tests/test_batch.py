import numpy as np
import pytest

from src.training.batch import (
    TensorBatchBuilder,
)
from src.training.dataset import (
    LanguageModelBatch,
)


def create_batch() -> LanguageModelBatch:
    return LanguageModelBatch(
        inputs=(
            (1, 2, 3),
            (2, 3, 4),
        ),
        targets=(
            (2, 3, 4),
            (3, 4, 5),
        ),
    )


def test_tensor_batch_builder_should_create_int64_arrays() -> None:
    result = TensorBatchBuilder().build(
        create_batch()
    )

    assert result.inputs.dtype == np.int64
    assert result.targets.dtype == np.int64


def test_tensor_batch_builder_should_create_expected_shapes() -> None:
    result = TensorBatchBuilder().build(
        create_batch()
    )

    assert result.inputs.shape == (
        2,
        3,
    )

    assert result.targets.shape == (
        2,
        3,
    )

    assert result.batch_size == 2
    assert result.context_size == 3


def test_tensor_batch_builder_should_preserve_values() -> None:
    result = TensorBatchBuilder().build(
        create_batch()
    )

    np.testing.assert_array_equal(
        result.inputs,
        np.array(
            [
                [1, 2, 3],
                [2, 3, 4],
            ],
            dtype=np.int64,
        ),
    )

    np.testing.assert_array_equal(
        result.targets,
        np.array(
            [
                [2, 3, 4],
                [3, 4, 5],
            ],
            dtype=np.int64,
        ),
    )


def test_tensor_batch_builder_should_reject_empty_batch() -> None:
    with pytest.raises(ValueError):
        TensorBatchBuilder().build(
            LanguageModelBatch(
                inputs=(),
                targets=(),
            )
        )


def test_tensor_batch_builder_should_reject_ragged_inputs() -> None:
    with pytest.raises(ValueError):
        TensorBatchBuilder().build(
            LanguageModelBatch(
                inputs=(
                    (1, 2, 3),
                    (2, 3),
                ),
                targets=(
                    (2, 3, 4),
                    (3, 4, 5),
                ),
            )
        )


def test_tensor_batch_builder_should_reject_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        TensorBatchBuilder().build(
            LanguageModelBatch(
                inputs=(
                    (1, 2, 3),
                ),
                targets=(
                    (2, 3),
                    ),
            )
        )
