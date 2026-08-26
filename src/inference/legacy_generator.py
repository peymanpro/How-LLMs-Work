from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.inference.sampling_strategy import (
    GreedySamplingStrategy,
    SamplingStrategy,
)
from src.inference.transformer_inference import (
    TransformerInference,
)


@dataclass(frozen=True)
class LegacyGenerationResult:
    token_ids: tuple[int, ...]


class LegacyTextGenerator:
    def __init__(
        self,
        inference: TransformerInference,
        context_size: int,
        strategy: SamplingStrategy | None = None,
    ) -> None:
        if context_size <= 0:
            raise ValueError(
                "Context size must be positive."
            )

        self._inference = inference
        self._context_size = context_size
        self._strategy = (
            strategy
            if strategy is not None
            else GreedySamplingStrategy()
        )

    @property
    def context_size(self) -> int:
        return self._context_size

    @property
    def inference(self) -> TransformerInference:
        return self._inference

    def generate(
        self,
        prompt: list[int],
        max_new_tokens: int,
        eos_token_id: int | None = None,
    ) -> LegacyGenerationResult:
        if not prompt:
            raise ValueError(
                "Prompt cannot be empty."
            )

        if max_new_tokens < 0:
            raise ValueError(
                "max_new_tokens cannot be negative."
            )

        if max_new_tokens == 0:
            return LegacyGenerationResult(
                token_ids=tuple(prompt)
            )

        if len(prompt) < self._context_size:
            raise ValueError(
                "Prompt must contain at least "
                f"{self._context_size} tokens."
            )

        generated = list(prompt)

        for _ in range(max_new_tokens):
            context = generated[
                -self._context_size:
            ]

            logits = self._inference.next_logits(
                context
            )

            token_id = self._strategy.sample(
                np.asarray(
                    logits,
                    dtype=np.float64,
                )
            )

            generated.append(
                token_id
            )

            if (
                eos_token_id is not None
                and token_id == eos_token_id
            ):
                break

        return LegacyGenerationResult(
            token_ids=tuple(generated)
        )
