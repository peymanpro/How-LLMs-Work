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
from src.training.transformer_session_client import (
    TransformerSessionClient,
)


@dataclass(frozen=True)
class GenerationStep:
    context: tuple[int, ...]
    prediction: NextTokenPrediction


@dataclass(frozen=True)
class SampledGenerationStep:
    context: tuple[int, ...]
    prediction: SamplingResult


class TransformerInference:
    def __init__(
        self,
        session: TransformerSessionClient,
        sampling_seed: int | None = None,
    ) -> None:
        self._session = session
        self._predictor = NextTokenPredictor()
        self._sampler = TemperatureSampler(
            seed=sampling_seed
        )

    def predict_next(
        self,
        token_ids: list[int],
    ) -> GenerationStep:
        logits = self._session.predict_next_token(
            token_ids
        )

        prediction = self._predictor.predict(
            logits
        )

        return GenerationStep(
            context=tuple(token_ids),
            prediction=prediction,
        )

    def sample_next(
        self,
        token_ids: list[int],
        temperature: float,
    ) -> SampledGenerationStep:
        logits = self._session.predict_next_token(
            token_ids
        )

        prediction = self._sampler.sample(
            logits[-1],
            temperature=temperature,
        )

        return SampledGenerationStep(
            context=tuple(token_ids),
            prediction=prediction,
        )
