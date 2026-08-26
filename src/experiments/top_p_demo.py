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

        for p in (
            0.5,
            0.9,
            1.0,
        ):
            result = (
                inference.sample_top_p_next(
                    context,
                    p=p,
                    temperature=1.0,
                )
            )

            print(
                f"Top-P = {p} → "
                f"{TOKENS[result.prediction.token_id]}"
            )

            print_distribution(
                f"Distribution @ Top-P={p}",
                result.prediction.probabilities.tolist(),
            )

    finally:
        session.close()


if __name__ == "__main__":
    main()
