from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LanguageModelExample:
    inputs: tuple[int, ...]
    targets: tuple[int, ...]


class LanguageModelDataset:
    def __init__(
        self,
        token_ids: list[int],
        context_size: int,
    ) -> None:
        if not token_ids:
            raise ValueError(
                "Token stream cannot be empty."
            )

        if context_size < 2:
            raise ValueError(
                "Context size must be at least 2."
            )

        if len(token_ids) < context_size + 1:
            raise ValueError(
                "Token stream is too short for the context size."
            )

        self._examples = self._build_examples(
            token_ids,
            context_size,
        )

    @staticmethod
    def _build_examples(
        token_ids: list[int],
        context_size: int,
    ) -> tuple[LanguageModelExample, ...]:
        examples: list[LanguageModelExample] = []

        for start in range(
            len(token_ids) - context_size,
        ):
            window = token_ids[
                start : start + context_size + 1
            ]

            examples.append(
                LanguageModelExample(
                    inputs=tuple(
                        window[:-1]
                    ),
                    targets=tuple(
                        window[1:]
                    ),
                )
            )

        return tuple(examples)

    @property
    def size(self) -> int:
        return len(self._examples)

    def get(
        self,
        index: int,
    ) -> LanguageModelExample:
        if not 0 <= index < self.size:
            raise IndexError(
                "Dataset index is out of range."
            )

        return self._examples[index]

    def examples(
        self,
    ) -> tuple[LanguageModelExample, ...]:
        return self._examples


@dataclass(frozen=True)
class LanguageModelBatch:
    inputs: tuple[tuple[int, ...], ...]
    targets: tuple[tuple[int, ...], ...]


class BatchBuilder:
    def create_batches(
        self,
        dataset: LanguageModelDataset,
        batch_size: int,
    ) -> tuple[LanguageModelBatch, ...]:
        if batch_size <= 0:
            raise ValueError(
                "Batch size must be positive."
            )

        batches: list[LanguageModelBatch] = []

        examples = dataset.examples()

        for start in range(
            0,
            len(examples),
            batch_size,
        ):
            batch_examples = examples[
                start : start + batch_size
            ]

            batches.append(
                LanguageModelBatch(
                    inputs=tuple(
                        example.inputs
                        for example in batch_examples
                    ),
                    targets=tuple(
                        example.targets
                        for example in batch_examples
                    ),
                )
            )

        return tuple(batches)
