from __future__ import annotations

from typing import Protocol


class GenerationBackend(Protocol):
    @property
    def context_size(self) -> int:
        ...

    def generate(
        self,
        prompt: list[int],
        max_new_tokens: int,
        eos_token_id: int | None = None,
    ) -> tuple[int, ...]:
        ...
