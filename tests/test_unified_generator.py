from src.inference.cached_generation_backend import (
    CachedGenerationBackend,
)
from src.inference.cached_generator import (
    CachedTextGenerator,
)
from src.inference.generation_backend import (
    GenerationBackend,
)
from src.inference.generator import (
    TextGenerator,
)
from src.inference.legacy_generation_backend import (
    LegacyGenerationBackend,
)
from src.inference.prefill_decode import (
    PrefillDecodeEngine,
)
from src.inference.unified_generator import (
    UnifiedTextGenerator,
)
from src.llm.cached_transformer_backbone import (
    CachedTransformerBackbone,
)
from src.llm.cached_transformer_language_model import (
    CachedTransformerLanguageModel,
)


class FakeLegacyInference:
    def next_logits(
        self,
        token_ids: list[int],
    ) -> list[float]:
        logits = [0.0] * 5
        logits[4] = 1.0
        return logits


def create_cached_generator() -> CachedTextGenerator:
    backbone = CachedTransformerBackbone(
        vocabulary_size=5,
        model_dimension=8,
        number_of_heads=2,
        context_size=8,
        seed=42,
    )

    model = CachedTransformerLanguageModel(
        backbone=backbone,
        vocabulary_size=5,
        seed=42,
    )

    return CachedTextGenerator(
        engine=PrefillDecodeEngine(
            model
        )
    )


def test_legacy_backend_should_implement_contract() -> None:
    generator = TextGenerator(
        inference=FakeLegacyInference(),
        context_size=4,
    )

    backend = LegacyGenerationBackend(
        generator=generator,
        context_size=4,
    )

    contract: GenerationBackend = backend

    result = contract.generate(
        prompt=[0, 1, 2, 3],
        max_new_tokens=1,
    )

    assert result == (
        0,
        1,
        2,
        3,
        4,
    )


def test_cached_backend_should_implement_contract() -> None:
    backend = CachedGenerationBackend(
        generator=create_cached_generator(),
        context_size=8,
    )

    contract: GenerationBackend = backend

    result = contract.generate(
        prompt=[0, 1, 2, 3],
        max_new_tokens=2,
    )

    assert len(result) == 6


def test_unified_generator_should_use_legacy_backend() -> None:
    generator = TextGenerator(
        inference=FakeLegacyInference(),
        context_size=4,
    )

    unified = UnifiedTextGenerator(
        LegacyGenerationBackend(
            generator=generator,
            context_size=4,
        )
    )

    result = unified.generate(
        prompt=[0, 1, 2, 3],
        max_new_tokens=1,
    )

    assert result == (
        0,
        1,
        2,
        3,
        4,
    )


def test_unified_generator_should_use_cached_backend() -> None:
    unified = UnifiedTextGenerator(
        CachedGenerationBackend(
            generator=create_cached_generator(),
            context_size=8,
        )
    )

    result = unified.generate(
        prompt=[0, 1, 2, 3],
        max_new_tokens=2,
    )

    assert len(result) == 6


def test_unified_generator_should_expose_backend_context_size() -> None:
    unified = UnifiedTextGenerator(
        CachedGenerationBackend(
            generator=create_cached_generator(),
            context_size=8,
        )
    )

    assert unified.context_size == 8
