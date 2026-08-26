from src.inference.cached_generation_backend import (
    CachedGenerationBackend,
)
from src.inference.cached_generator import (
    CachedTextGenerator,
)
from src.inference.prefill_decode import (
    PrefillDecodeEngine,
)
from src.llm.cached_transformer_backbone import (
    CachedTransformerBackbone,
)
from src.llm.cached_transformer_language_model import (
    CachedTransformerLanguageModel,
)


def create_generator(
    context_size: int,
) -> CachedTextGenerator:
    backbone = CachedTransformerBackbone(
        vocabulary_size=5,
        model_dimension=8,
        number_of_heads=2,
        context_size=context_size,
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


def test_cached_generator_should_derive_context_size_from_model() -> None:
    generator = create_generator(
        context_size=6
    )

    assert generator.context_size == 6


def test_cached_backend_should_not_duplicate_context_configuration() -> None:
    generator = create_generator(
        context_size=6
    )

    backend = CachedGenerationBackend(
        generator=generator,
    )

    assert backend.context_size == 6
