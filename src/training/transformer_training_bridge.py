from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.training.transformer_session_client import (
    TransformerSessionClient,
)


@dataclass(frozen=True)
class ExternalTrainingResult:
    initial_loss: float
    final_loss: float
    epochs: int
    sequences: int


class HowTransformersWorkTrainingBridge:
    def __init__(
        self,
        repository_root: Path,
    ) -> None:
        self._repository_root = repository_root

    def train(
        self,
        *,
        vocabulary_size: int,
        model_dimension: int,
        head_dimension: int,
        feed_forward_dimension: int,
        maximum_sequence_length: int,
        learning_rate: float,
        epochs: int,
        sequences: list[list[int]],
        targets: list[list[int]],
    ) -> ExternalTrainingResult:
        if not sequences:
            raise ValueError(
                "Training sequences cannot be empty."
            )

        if len(sequences) != len(targets):
            raise ValueError(
                "Sequences and targets must have equal length."
            )

        client = TransformerSessionClient(
            self._repository_root
        )

        try:
            client.initialize(
                vocabulary_size=vocabulary_size,
                model_dimension=model_dimension,
                head_dimension=head_dimension,
                feed_forward_dimension=feed_forward_dimension,
                maximum_sequence_length=maximum_sequence_length,
                learning_rate=learning_rate,
            )

            initial_loss = client.evaluate(
                sequences=sequences,
                targets=targets,
            )

            final = client.train_many(
                sequences=sequences,
                targets=targets,
                epochs=epochs,
            )

            return ExternalTrainingResult(
                initial_loss=initial_loss,
                final_loss=final.loss,
                epochs=epochs,
                sequences=len(sequences),
            )
        finally:
            client.close()
