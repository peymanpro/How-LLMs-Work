from __future__ import annotations

from dataclasses import dataclass

from src.inference.transformer_inference import (
    TransformerInference,
)


@dataclass(frozen=True)
class GenerationResult:
    token_ids: tuple[int, ...]


class GreedyTextGenerator:
    def __init__(
        self,
        inference: TransformerInference,
        context_size: int,
    ) -> None:
        if context_size <= 0:
            raise ValueError(
                "Context size must be positive."
            )

        self._inference = inference
        self._context_size = context_size

    def generate(
        self,
        prompt: list[int],
        max_new_tokens: int,
        eos_token_id: int | None = None,
    ) -> GenerationResult:
        if not prompt:
            raise ValueError(
                "Prompt cannot be empty."
            )

        if max_new_tokens < 0:
            raise ValueError(
                "max_new_tokens cannot be negative."
            )

        if max_new_tokens == 0:
            return GenerationResult(
                token_ids=tuple(prompt)
            )

        if len(prompt) < self._context_size:
            raise ValueError(
                "Prompt must contain at least "
                f"{self._context_size} tokens for the current "
                "Transformer architecture."
            )

        generated = list(prompt)

        for _ in range(max_new_tokens):
            context = generated[
                -self._context_size:
            ]

            prediction = (
                self._inference.predict_next(
                    context
                )
            )

            next_token = (
                prediction.prediction.token_id
            )

            generated.append(
                next_token
            )

            if (
                eos_token_id is not None
                and next_token == eos_token_id
            ):
                break

        return GenerationResult(
            token_ids=tuple(generated)
        )

