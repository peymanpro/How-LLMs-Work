from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.inference.prefill_decode import (
    PrefillDecodeEngine,
)
from src.inference.sampling_strategy import (
    GreedySamplingStrategy,
    SamplingStrategy,
)


@dataclass(frozen=True)
class CachedGenerationResult:
    token_ids: tuple[int, ...]


class CachedTextGenerator:
    def __init__(
        self,
        engine: PrefillDecodeEngine,
        strategy: SamplingStrategy | None = None,
    ) -> None:
        self._engine = engine
        self._strategy = (
            strategy
            if strategy is not None
            else GreedySamplingStrategy()
        )

    @property
    def context_size(self) -> int:
        return self._engine.context_size

    def generate(
        self,
        prompt: list[int],
        max_new_tokens: int,
        eos_token_id: int | None = None,
    ) -> CachedGenerationResult:
        if not prompt:
            raise ValueError(
                "Prompt cannot be empty."
            )

        if max_new_tokens < 0:
            raise ValueError(
                "max_new_tokens cannot be negative."
            )

        if len(prompt) > self._engine.context_size:
            raise ValueError(
                "Prompt exceeds context size."
            )

        generated = list(prompt)

        if max_new_tokens == 0:
            return CachedGenerationResult(
                token_ids=tuple(generated)
            )

        prefill = self._engine.prefill(
            prompt
        )

        logits = prefill.next_logits

        for _ in range(max_new_tokens):
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

            if (
                self._engine.cache_length
                >= self._engine.context_size
            ):
                window = generated[
                    -self._engine.context_size :
                ]

                prefill = self._engine.prefill(
                    window
                )

                logits = prefill.next_logits

            else:
                decoded = self._engine.decode(
                    token_id
                )

                logits = decoded.next_logits

        return CachedGenerationResult(
            token_ids=tuple(generated)
        )

