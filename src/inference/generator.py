from __future__ import annotations

from dataclasses import dataclass

from src.inference.generation_backend import (
    GenerationBackend,
)
from src.inference.legacy_generation_backend import (
    LegacyGenerationBackend,
)
from src.inference.legacy_generator import (
    LegacyTextGenerator,
)
from src.inference.sampling_strategy import (
    SamplingStrategy,
)
from src.inference.transformer_inference import (
    TransformerInference,
)


@dataclass(frozen=True)
class GenerationResult:
    token_ids: tuple[int, ...]


class TextGenerator:
    def __init__(
        self,
        backend: GenerationBackend | None = None,
        *,
        inference: TransformerInference | None = None,
        context_size: int | None = None,
        strategy: SamplingStrategy | None = None,
    ) -> None:
        if backend is not None:
            if (
                inference is not None
                or context_size is not None
                or strategy is not None
            ):
                raise ValueError(
                    "backend cannot be combined with legacy arguments."
                )

            self._backend = backend
            self._legacy_generator: LegacyTextGenerator | None = None
            return

        if inference is None:
            raise ValueError(
                "Either backend or inference must be provided."
            )

        if context_size is None:
            raise ValueError(
                "context_size is required with inference."
            )

        legacy_generator = LegacyTextGenerator(
            inference=inference,
            context_size=context_size,
            strategy=strategy,
        )

        self._legacy_generator = legacy_generator
        self._backend = LegacyGenerationBackend(
            generator=legacy_generator,
        )

    @property
    def context_size(self) -> int:
        return self._backend.context_size

    @property
    def legacy_inference(
        self,
    ) -> TransformerInference | None:
        if self._legacy_generator is None:
            return None

        return self._legacy_generator.inference

    def generate(
        self,
        prompt: list[int],
        max_new_tokens: int,
        eos_token_id: int | None = None,
    ) -> GenerationResult:
        token_ids = self._backend.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
        )

        return GenerationResult(
            token_ids=token_ids
        )


GreedyTextGenerator = TextGenerator
