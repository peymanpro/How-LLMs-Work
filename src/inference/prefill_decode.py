from __future__ import annotations

from dataclasses import dataclass

from src.llm.cached_transformer_language_model import (
    CachedTransformerLanguageModel,
)


@dataclass(frozen=True)
class PrefillResult:
    token_ids: tuple[int, ...]
    next_logits: list[float]
    cache_length: int


@dataclass(frozen=True)
class DecodeResult:
    token_id: int
    next_logits: list[float]
    cache_length: int


class PrefillDecodeEngine:
    def __init__(
        self,
        model: CachedTransformerLanguageModel,
    ) -> None:
        self._model = model

    @property
    def context_size(self) -> int:
        return self._model.context_size

    @property
    def cache_length(self) -> int:
        return self._model.cache_length

    def reset(self) -> None:
        self._model.reset_cache()

    def prefill(
        self,
        token_ids: list[int],
    ) -> PrefillResult:
        if not token_ids:
            raise ValueError(
                "Prompt cannot be empty."
            )

        if len(token_ids) > self.context_size:
            raise ValueError(
                "Prompt exceeds context size."
            )

        self._model.reset_cache()

        logits: list[float] = []

        for token_id in token_ids:
            logits = (
                self._model.incremental_logits(
                    token_id
                ).tolist()
            )

        return PrefillResult(
            token_ids=tuple(token_ids),
            next_logits=logits,
            cache_length=self.cache_length,
        )

    def decode(
        self,
        token_id: int,
    ) -> DecodeResult:
        if self.cache_length >= self.context_size:
            raise ValueError(
                "Context size has been exceeded. "
                "Use prefill() with the current context window."
            )

        logits = (
            self._model.incremental_logits(
                token_id
            ).tolist()
        )

        return DecodeResult(
            token_id=token_id,
            next_logits=logits,
            cache_length=self.cache_length,
        )
