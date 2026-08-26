from __future__ import annotations

from src.inference.cached_generator import (
    CachedTextGenerator,
)
from src.inference.generation_backend import (
    GenerationBackend,
)


class CachedGenerationBackend:
    def __init__(
        self,
        generator: CachedTextGenerator,
        context_size: int | None = None,
    ) -> None:
        if context_size is not None and context_size <= 0:
            raise ValueError(
                "context_size must be positive."
            )

        self._generator = generator

    @property
    def context_size(self) -> int:
        return self._generator.context_size

    def generate(
        self,
        prompt: list[int],
        max_new_tokens: int,
        eos_token_id: int | None = None,
    ) -> tuple[int, ...]:
        result = self._generator.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
        )

        return result.token_ids


def assert_generation_backend_contract(
    backend: CachedGenerationBackend,
) -> GenerationBackend:
    return backend
