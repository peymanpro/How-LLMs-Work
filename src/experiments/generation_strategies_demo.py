from __future__ import annotations

from pathlib import Path

from src.inference.generator import (
    TextGenerator,
)
from src.inference.sampling_strategy import (
    GreedySamplingStrategy,
    SamplingStrategy,
    TemperatureSamplingStrategy,
    TopKSamplingStrategy,
    TopPSamplingStrategy,
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


def train_model(
    session: TransformerSessionClient,
) -> None:
    context = [
        0,
        1,
        2,
        3,
    ]

    targets = [
        1,
        2,
        3,
        4,
    ]

    for _ in range(300):
        session.train(
            context,
            targets,
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

        train_model(
            session
        )

        inference = TransformerInference(
            session,
            sampling_seed=42,
        )

        prompt = [
            0,
            1,
            2,
            3,
        ]

        strategies: list[
            tuple[str, SamplingStrategy]
        ] = [
            (
                "Greedy",
                GreedySamplingStrategy(),
            ),
            (
                "Temperature = 1.0",
                TemperatureSamplingStrategy(
                    temperature=1.0,
                    seed=42,
                ),
            ),
            (
                "Top-K = 2",
                TopKSamplingStrategy(
                    k=2,
                    temperature=1.0,
                    seed=42,
                ),
            ),
            (
                "Top-P = 0.9",
                TopPSamplingStrategy(
                    p=0.9,
                    temperature=1.0,
                    seed=42,
                ),
            ),
        ]

        print("HowLLMsWork")
        print("============")
        print()
        print(
            "Prompt: "
            + decode(
                tuple(prompt)
            )
        )
        print()

        for name, strategy in strategies:
            generator = TextGenerator(
                inference=inference,
                context_size=4,
                strategy=strategy,
            )

            result = generator.generate(
                prompt=prompt,
                max_new_tokens=3,
                eos_token_id=4,
            )

            print(
                f"{name}:"
            )
            print(
                f"  {decode(result.token_ids)}"
            )
            print()

    finally:
        session.close()


if __name__ == "__main__":
    main()


