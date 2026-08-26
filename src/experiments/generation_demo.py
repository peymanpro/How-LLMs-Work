from __future__ import annotations

from pathlib import Path

from src.inference.generator import (
    GreedyTextGenerator,
)
from src.inference.transformer_inference import (
    TransformerInference,
)
from src.training.transformer_session_client import (
    TransformerSessionClient,
)

TOKENS = {
    0: "the",
    1: "cat",
    2: "drinks",
    3: "milk",
    4: "<eos>",
}


def decode(
    token_ids: tuple[int, ...],
) -> str:
    return " ".join(
        TOKENS[token_id]
        for token_id in token_ids
    )


def main() -> None:
    repository = (
        Path(__file__).resolve().parents[3]
        / "HowTransformersWork"
    )

    session = TransformerSessionClient(
        repository
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

        training_context = [
            0,
            1,
            2,
            3,
        ]

        training_targets = [
            1,
            2,
            3,
            4,
        ]

        for _ in range(300):
            session.train(
                training_context,
                training_targets,
            )

        inference = TransformerInference(
            session
        )

        generator = GreedyTextGenerator(
            inference=inference,
            context_size=4,
        )

        result = generator.generate(
            prompt=[0, 1, 2, 3],
            max_new_tokens=3,
            eos_token_id=4,
        )

        print("HowLLMsWork")
        print("============")
        print()
        print(
            f"Generated IDs: {result.token_ids}"
        )
        print()
        print(
            f"Generated text: {decode(result.token_ids)}"
        )

    finally:
        session.close()


if __name__ == "__main__":
    main()

