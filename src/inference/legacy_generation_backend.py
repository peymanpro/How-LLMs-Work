from __future__ import annotations

from typing import Protocol

from src.inference.generation_backend import (
    GenerationBackend,
)


class GenerationResultProtocol(Protocol):
    @property
    def token_ids(self) -> tuple[int, ...]:
        ...


class GeneratorDelegate(Protocol):
    @property
    def context_size(self) -> int:
        ...

    def generate(
        self,
        prompt: list[int],
        max_new_tokens: int,
        eos_token_id: int | None = None,
    ) -> GenerationResultProtocol:
        ...


class LegacyGenerationBackend:
    def __init__(
        self,
        generator: GeneratorDelegate,
        context_size: int | None = None,
    ) -> None:
        self._generator = generator
        self._context_size = context_size

    @property
    def context_size(self) -> int:
        if self._context_size is not None:
            return self._context_size

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
    backend: LegacyGenerationBackend,
) -> GenerationBackend:
    return backend
