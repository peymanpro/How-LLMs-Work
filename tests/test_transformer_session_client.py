from pathlib import Path

import pytest

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
def test_transformer_session_should_keep_model_alive() -> None:
    client = TransformerSessionClient(
        repository_root()
    )

    try:
        client.initialize(
            vocabulary_size=5,
            model_dimension=8,
            head_dimension=4,
            head_focuses=[0, 1],
            feed_forward_dimension=16,
            maximum_sequence_length=4,
            learning_rate=0.05,
        )

        before = client.forward(
            [0, 1, 2, 3]
        )

        first = client.train(
            [0, 1, 2, 3],
            [1, 2, 3, 4],
        )

        second = client.train(
            [0, 1, 2, 3],
            [1, 2, 3, 4],
        )

        after = client.forward(
            [0, 1, 2, 3]
        )

        assert first.loss != second.loss
        assert before.logits != after.logits

    finally:
        client.close()
