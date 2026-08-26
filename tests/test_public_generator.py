from src.inference.cached_generation_backend import (
    CachedGenerationBackend,
)
from src.inference.cached_generator import (
    CachedTextGenerator,
)
from src.inference.generator import (
    TextGenerator,
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


def create_generator() -> TextGenerator:
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

    engine = PrefillDecodeEngine(
        model
    )

    backend = CachedGenerationBackend(
        generator=CachedTextGenerator(
            engine
        ),
        context_size=8,
    )

    return TextGenerator(
        backend
    )


def test_public_text_generator_should_use_cached_backend() -> None:
    generator = create_generator()

    result = generator.generate(
        prompt=[0, 1, 2, 3],
        max_new_tokens=3,
    )

    assert len(
        result.token_ids
    ) == 7


def test_public_text_generator_should_expose_context_size() -> None:
    generator = create_generator()

    assert generator.context_size == 8
