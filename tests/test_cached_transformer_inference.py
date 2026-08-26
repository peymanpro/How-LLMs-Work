from src.inference.cached_transformer_inference import (
    CachedTransformerInference,
)
from src.llm.cached_transformer_backbone import (
    CachedTransformerBackbone,
)
from src.llm.cached_transformer_language_model import (
    CachedTransformerLanguageModel,
)


def create_inference() -> CachedTransformerInference:
    backbone = CachedTransformerBackbone(
        vocabulary_size=5,
        model_dimension=8,
        number_of_heads=2,
        context_size=4,
        seed=42,
    )

    model = CachedTransformerLanguageModel(
        backbone=backbone,
        vocabulary_size=5,
        seed=42,
    )

    return CachedTransformerInference(
        model=model,
        sampling_seed=42,
    )


def test_cached_inference_should_return_next_logits() -> None:
    inference = create_inference()

    logits = inference.next_logits(
        0
    )

    assert len(logits) == 5
    assert inference.cache_length == 1


def test_cached_inference_should_accumulate_cache() -> None:
    inference = create_inference()

    inference.next_logits(0)
    inference.next_logits(1)
    inference.next_logits(2)

    assert inference.cache_length == 3


def test_cached_inference_should_reset_cache() -> None:
    inference = create_inference()

    inference.next_logits(0)
    inference.next_logits(1)

    inference.reset()

    assert inference.cache_length == 0


def test_cached_inference_should_sample_next_token() -> None:
    inference = create_inference()

    result = inference.sample_next(
        0,
        temperature=1.0,
    )

    assert 0 <= result.prediction.token_id < 5


def test_cached_inference_should_sample_with_top_k() -> None:
    inference = create_inference()

    result = inference.sample_top_k_next(
        0,
        k=2,
    )

    assert 0 <= result.prediction.token_id < 5


def test_cached_inference_should_sample_with_top_p() -> None:
    inference = create_inference()

    result = inference.sample_top_p_next(
        0,
        p=0.9,
    )

    assert 0 <= result.prediction.token_id < 5
