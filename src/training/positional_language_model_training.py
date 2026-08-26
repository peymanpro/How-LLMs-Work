from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.llm.positional_language_model import (
    PositionalContextLanguageModel,
)
from src.training.language_model_objective import (
    LanguageModelObjective,
)


@dataclass(frozen=True)
class PositionalTrainingStepResult:
    loss: float
    perplexity: float


class PositionalLanguageModelTrainingStep:
    def __init__(
        self,
        learning_rate: float,
    ) -> None:
        if learning_rate <= 0.0:
            raise ValueError(
                "Learning rate must be positive."
            )

        self._learning_rate = learning_rate
        self._objective = (
            LanguageModelObjective()
        )

    def run(
        self,
        model: PositionalContextLanguageModel,
        token_ids: list[int],
        targets: list[int],
    ) -> PositionalTrainingStepResult:
        forward_result = model.forward(
            token_ids
        )

        target_array = np.asarray(
            targets,
            dtype=np.int64,
        )

        if target_array.shape != (
            model.context_size,
        ):
            raise ValueError(
                "Target length must match context size."
            )

        loss = self._objective.cross_entropy(
            forward_result.logits[None, ...],
            target_array[None, ...],
        )

        output_gradient = (
            self._cross_entropy_gradient(
                forward_result.logits,
                target_array,
            )
        )

        gradients = model.backward(
            token_ids=token_ids,
            forward_result=forward_result,
            output_gradient=output_gradient,
        )

        model.apply_gradients(
            gradients=gradients,
            learning_rate=self._learning_rate,
        )

        return PositionalTrainingStepResult(
            loss=loss,
            perplexity=self._objective.perplexity(
                loss
            ),
        )

    @staticmethod
    def _cross_entropy_gradient(
        logits: np.ndarray,
        targets: np.ndarray,
    ) -> np.ndarray:
        shifted = (
            logits
            - np.max(
                logits,
                axis=1,
                keepdims=True,
            )
        )

        probabilities = np.exp(
            shifted
        )

        probabilities /= np.sum(
            probabilities,
            axis=1,
            keepdims=True,
        )

        gradient = probabilities.copy()

        gradient[
            np.arange(
                targets.shape[0]
            ),
            targets,
        ] -= 1.0

        gradient /= targets.shape[0]

        return gradient
