from pathlib import Path

import pytest

from src.inference.transformer_inference import (
    TransformerInference,
)
from src.training.transformer_session_client import (
    TransformerSessionClient,
)


def repository_root() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "HowTransformersWork"
    )


@pytest.mark.skipif(
    not repository_root().exists(),
    reason="HowTransformersWork repository is not available.",
)
def test_transformer_inference_should_return_next_token_prediction() -> None:
    session = TransformerSessionClient(
        repository_root()
    )

    try:
        session.initialize(
            vocabulary_size=5,
            model_dimension=8,
            head_dimension=4,
            head_focuses=[0, 1],
            feed_forward_dimension=16,
            maximum_sequence_length=4,
            learning_rate=0.05,
        )

        inference = TransformerInference(
            session
        )

        result = inference.predict_next(
            [0, 1, 2, 3]
        )

        assert len(
            result.prediction.probabilities
        ) == 5

        assert 0 <= result.prediction.token_id < 5

        assert sum(
            result.prediction.probabilities
        ) == pytest.approx(1.0)

    finally:
        session.close()
def test_transformer_inference_should_sample_next_token() -> None:
    session = TransformerSessionClient(
        repository_root()
    )

    try:
        session.initialize(
            vocabulary_size=5,
            model_dimension=8,
            head_dimension=4,
            head_focuses=[0, 1],
            feed_forward_dimension=16,
            maximum_sequence_length=4,
            learning_rate=0.05,
        )

        inference = TransformerInference(
            session,
            sampling_seed=42,
        )

        result = inference.sample_next(
            [0, 1, 2, 3],
            temperature=1.0,
        )

        assert 0 <= (
            result.prediction.token_id
        ) < 5

        assert sum(
            result.prediction.probabilities
        ) == pytest.approx(1.0)

    finally:
        session.close()
def test_transformer_inference_should_sample_with_top_k() -> None:
    session = TransformerSessionClient(
        repository_root()
    )

    try:
        session.initialize(
            vocabulary_size=5,
            model_dimension=8,
            head_dimension=4,
            head_focuses=[0, 1],
            feed_forward_dimension=16,
            maximum_sequence_length=4,
            learning_rate=0.05,
        )

        inference = TransformerInference(
            session,
            sampling_seed=42,
        )

        result = inference.sample_top_k_next(
            [0, 1, 2, 3],
            k=2,
            temperature=1.0,
        )

        probabilities = (
            result.prediction.probabilities
        )

        assert sum(
            probability > 0.0
            for probability in probabilities
        ) == 2

        assert 0 <= (
            result.prediction.token_id
        ) < 5

    finally:
        session.close()
