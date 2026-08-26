from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.llm.language_model import LanguageModel
from src.training.dataset import LanguageModelBatch
from src.training.evaluation import LanguageModelEvaluator


@dataclass(frozen=True)
class ModelEvaluationResult:
    loss: float
    perplexity: float
    examples: int


class BatchLanguageModelEvaluator:
    def __init__(self) -> None:
        self._evaluator = LanguageModelEvaluator()

    def evaluate_batch(
        self,
        model: LanguageModel,
        batch: LanguageModelBatch,
    ) -> ModelEvaluationResult:
        if not batch.inputs:
            raise ValueError(
                "Batch cannot be empty."
            )

        if len(batch.inputs) != len(batch.targets):
            raise ValueError(
                "Batch inputs and targets must have equal length."
            )

        logits_per_example: list[np.ndarray] = []

        for inputs in batch.inputs:
            logits = model.logits(list(inputs))

            if logits.ndim != 2:
                raise ValueError(
                    "Model logits must have shape "
                    "(sequence_length, vocabulary_size)."
                )

            if logits.shape[0] != len(inputs):
                raise ValueError(
                    "Model logits sequence length does not "
                    "match the input sequence length."
                )

            if logits.shape[1] != model.vocabulary_size:
                raise ValueError(
                    "Model vocabulary dimension does not "
                    "match model.vocabulary_size."
                )

            if len(inputs) != model.context_size:
                raise ValueError(
                    "Input sequence length does not "
                    "match model.context_size."
                )

            logits_per_example.append(logits)

        logits_batch = np.stack(
            logits_per_example,
            axis=0,
        )

        targets_batch = np.asarray(
            batch.targets,
            dtype=np.int64,
        )

        result = self._evaluator.evaluate(
            logits_batch,
            targets_batch,
        )

        return ModelEvaluationResult(
            loss=result.loss,
            perplexity=result.perplexity,
            examples=len(batch.inputs),
        )
