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


def print_distribution(
    title: str,
    probabilities: list[float],
) -> None:
    print(title)
    print("-" * len(title))

    for token_id, probability in enumerate(
        probabilities
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

        context = [
            0,
            1,
            2,
            3,
        ]

        for _ in range(300):
            session.train(
                context,
                [1, 2, 3, 4],
            )

        inference = TransformerInference(
            session,
            sampling_seed=42,
        )

        cold = inference.sample_next(
            context,
            temperature=0.2,
        )

        normal = inference.sample_next(
            context,
            temperature=1.0,
        )

        hot = inference.sample_next(
            context,
            temperature=2.0,
        )

        print("HowLLMsWork")
        print("============")
        print()
        print(
            "Context: the cat drinks milk"
        )
        print()

        print(
            f"Temperature 0.2 → "
            f"{TOKENS[cold.prediction.token_id]}"
        )

        print_distribution(
            "Distribution @ T=0.2",
            cold.prediction.probabilities.tolist(),
        )

        print(
            f"Temperature 1.0 → "
            f"{TOKENS[normal.prediction.token_id]}"
        )

        print_distribution(
            "Distribution @ T=1.0",
            normal.prediction.probabilities.tolist(),
        )

        print(
            f"Temperature 2.0 → "
            f"{TOKENS[hot.prediction.token_id]}"
        )

        print_distribution(
            "Distribution @ T=2.0",
            hot.prediction.probabilities.tolist(),
        )

    finally:
        session.close()


if __name__ == "__main__":
    main()
