from __future__ import annotations

from dataclasses import dataclass

from src.inference.next_token import (
    NextTokenPrediction,
    NextTokenPredictor,
)
from src.inference.sampling import (
    SamplingResult,
    TemperatureSampler,
)
from src.inference.top_k_sampling import (
    TopKSampler,
    TopKSamplingResult,
)
from src.inference.top_p_sampling import (
    TopPSampler,
    TopPSamplingResult,
)
from src.llm.cached_transformer_language_model import (
    CachedTransformerLanguageModel,
)


@dataclass(frozen=True)
class CachedGenerationStep:
    token_id: int
    prediction: NextTokenPrediction


@dataclass(frozen=True)
class CachedSampledGenerationStep:
    token_id: int
    prediction: SamplingResult


@dataclass(frozen=True)
class CachedTopKGenerationStep:
    token_id: int
    prediction: TopKSamplingResult


@dataclass(frozen=True)
class CachedTopPGenerationStep:
    token_id: int
    prediction: TopPSamplingResult


class CachedTransformerInference:
    def __init__(
        self,
        model: CachedTransformerLanguageModel,
        sampling_seed: int | None = None,
    ) -> None:
        self._model = model
        self._predictor = NextTokenPredictor()

        self._sampler = TemperatureSampler(
            seed=sampling_seed
        )

        self._top_k_sampler = TopKSampler(
            seed=sampling_seed
        )

        self._top_p_sampler = TopPSampler(
            seed=sampling_seed
        )

    @property
    def cache_length(self) -> int:
        return self._model.cache_length

    def reset(self) -> None:
        self._model.reset_cache()

    def next_logits(
        self,
        token_id: int,
    ) -> list[float]:
        logits = self._model.incremental_logits(
            token_id
        )

        return logits.tolist()

    def predict_next(
        self,
        token_id: int,
    ) -> CachedGenerationStep:
        logits = self._model.incremental_logits(
            token_id
        )

        prediction = self._predictor.predict(
            [logits.tolist()]
        )

        return CachedGenerationStep(
            token_id=token_id,
            prediction=prediction,
        )

    def sample_next(
        self,
        token_id: int,
        temperature: float,
    ) -> CachedSampledGenerationStep:
        logits = self.next_logits(
            token_id
        )

        prediction = self._sampler.sample(
            logits,
            temperature=temperature,
        )

        return CachedSampledGenerationStep(
            token_id=token_id,
            prediction=prediction,
        )

    def sample_top_k_next(
        self,
        token_id: int,
        k: int,
        temperature: float = 1.0,
    ) -> CachedTopKGenerationStep:
        logits = self.next_logits(
            token_id
        )

        prediction = self._top_k_sampler.sample(
            logits,
            k=k,
        )

        return CachedTopKGenerationStep(
            token_id=token_id,
            prediction=prediction,
        )

    def sample_top_p_next(
        self,
        token_id: int,
        p: float,
        temperature: float = 1.0,
    ) -> CachedTopPGenerationStep:
        logits = self.next_logits(
            token_id
        )

        prediction = self._top_p_sampler.sample(
            logits,
            p=p,
            temperature=temperature,
        )

        return CachedTopPGenerationStep(
            token_id=token_id,
            prediction=prediction,
        )
