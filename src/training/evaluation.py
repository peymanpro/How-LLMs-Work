from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.training.language_model_objective import (
    LanguageModelObjective,
)
from src.training.metrics import LanguageModelMetrics


@dataclass(frozen=True)
class EvaluationResult:
    loss: float
    perplexity: float
    token_accuracy: float


class LanguageModelEvaluator:
    def __init__(self) -> None:
        self._objective = LanguageModelObjective()
        self._metrics = LanguageModelMetrics()

    def evaluate(
        self,
        logits: np.ndarray,
        targets: np.ndarray,
    ) -> EvaluationResult:
        loss = self._objective.cross_entropy(
            logits,
            targets,
        )

        return EvaluationResult(
            loss=loss,
            perplexity=self._objective.perplexity(
                loss
            ),
            token_accuracy=self._metrics.token_accuracy(
                logits,
                targets,
            ),
        )
