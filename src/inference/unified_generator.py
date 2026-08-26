from __future__ import annotations

from src.inference.generation_backend import (
    GenerationBackend,
)


class UnifiedTextGenerator:
    def __init__(
        self,
        backend: GenerationBackend,
    ) -> None:
        self._backend = backend

    @property
    def context_size(self) -> int:
        return self._backend.context_size

    def generate(
        self,
        prompt: list[int],
        max_new_tokens: int,
        eos_token_id: int | None = None,
    ) -> tuple[int, ...]:
        return self._backend.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
        )
