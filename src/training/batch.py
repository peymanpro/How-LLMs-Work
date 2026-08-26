from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.training.dataset import LanguageModelBatch


@dataclass(frozen=True)
class TensorBatch:
    inputs: np.ndarray
    targets: np.ndarray

    @property
    def batch_size(self) -> int:
        return self.inputs.shape[0]

    @property
    def context_size(self) -> int:
        return self.inputs.shape[1]


class TensorBatchBuilder:
    def build(
        self,
        batch: LanguageModelBatch,
    ) -> TensorBatch:
        if not batch.inputs:
            raise ValueError(
                "Batch cannot be empty."
            )

        if len(batch.inputs) != len(
            batch.targets
        ):
            raise ValueError(
                "Inputs and targets must have equal batch size."
            )

        input_lengths = {
            len(sequence)
            for sequence in batch.inputs
        }

        target_lengths = {
            len(sequence)
            for sequence in batch.targets
        }

        if len(input_lengths) != 1:
            raise ValueError(
                "All input sequences must have equal length."
            )

        if len(target_lengths) != 1:
            raise ValueError(
                "All target sequences must have equal length."
            )

        if input_lengths != target_lengths:
            raise ValueError(
                "Input and target sequence lengths must match."
            )

        inputs = np.asarray(
            batch.inputs,
            dtype=np.int64,
        )

        targets = np.asarray(
            batch.targets,
            dtype=np.int64,
        )

        return TensorBatch(
            inputs=inputs,
            targets=targets,
        )
