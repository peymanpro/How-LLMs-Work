from __future__ import annotations

from src.inference.generation_backend import (
    GenerationBackend,
)
from src.inference.generator import (
    TextGenerator,
)


class LegacyGenerationBackend:
    def __init__(
        self,
        generator: TextGenerator,
        context_size: int,
    ) -> None:
        self._generator = generator
        self._context_size = context_size

    @property
    def context_size(self) -> int:
        return self._context_size

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
    backend: LegacyGenerationBackend,
) -> GenerationBackend:
    return backend
