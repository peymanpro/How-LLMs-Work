from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


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
        head_focuses: list[int],
        feed_forward_dimension: int,
        maximum_sequence_length: int,
        learning_rate: float,
        epochs: int,
        sequences: list[list[int]],
        targets: list[list[int]],
    ) -> ExternalTrainingResult:
        payload = {
            "vocabulary_size": vocabulary_size,
            "model_dimension": model_dimension,
            "head_dimension": head_dimension,
            "head_focuses": head_focuses,
            "feed_forward_dimension": feed_forward_dimension,
            "maximum_sequence_length": (
                maximum_sequence_length
            ),
            "learning_rate": learning_rate,
            "epochs": epochs,
            "sequences": sequences,
            "targets": targets,
        }

        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.experiments.train_language_model",
            ],
            cwd=self._repository_root,
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            check=True,
        )

        result = json.loads(
            completed.stdout
        )

        return ExternalTrainingResult(
            initial_loss=float(
                result["initial_loss"]
            ),
            final_loss=float(
                result["final_loss"]
            ),
            epochs=int(
                result["epochs"]
            ),
            sequences=int(
                result["sequences"]
            ),
        )
