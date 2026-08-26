from __future__ import annotations

from pathlib import Path

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


def print_prediction(
    title: str,
    result,
) -> None:
    predicted = TOKENS[
        result.prediction.token_id
    ]

    print(title)
    print("-" * len(title))
    print(
        f"Predicted token: {predicted}"
    )

    print(
        "Probabilities:"
    )

    for token_id, probability in enumerate(
        result.prediction.probabilities
    ):
        print(
            f"  {TOKENS[token_id]:>7}: "
            f"{probability:.6f}"
        )

    print()


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

        inference = TransformerInference(
            session
        )

        context = [0, 1, 2, 3]
        targets = [1, 2, 3, 4]

        before = inference.predict_next(
            context
        )

        for _ in range(200):
            session.train(
                context,
                targets,
            )

        after = inference.predict_next(
            context
        )

        print("HowLLMsWork")
        print("============")
        print()
        print(
            "Context: the cat drinks milk"
        )
        print()

        print_prediction(
            "Before Training",
            before,
        )

        print_prediction(
            "After Training",
            after,
        )

    finally:
        session.close()


if __name__ == "__main__":
    main()
