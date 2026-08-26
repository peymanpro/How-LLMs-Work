from __future__ import annotations

from pathlib import Path

from src.training.transformer_session_client import (
    TransformerSessionClient,
)


def main() -> None:
    repository = (
        Path(__file__).resolve().parents[3]
        / "HowTransformersWork"
    )

    client = TransformerSessionClient(
        repository
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

        first_loss = client.train(
            [0, 1, 2, 3],
            [1, 2, 3, 4],
        ).loss

        final_loss = first_loss

        for _ in range(99):
            final_loss = client.train(
                [0, 1, 2, 3],
                [1, 2, 3, 4],
            ).loss

        after = client.forward(
            [0, 1, 2, 3]
        )

        print("HowLLMsWork")
        print("============")
        print()
        print(
            "Persistent Transformer Session"
        )
        print()
        print(
            f"Initial step loss: {first_loss:.6f}"
        )
        print(
            f"Final step loss:   {final_loss:.6f}"
        )
        print()
        print(
            "Output changed:"
        )
        print(
            before.logits != after.logits
        )

    finally:
        client.close()


if __name__ == "__main__":
    main()
