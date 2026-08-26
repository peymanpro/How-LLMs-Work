from __future__ import annotations

from src.llm.simple_language_model import (
    SimpleContextLanguageModel,
)
from src.training.language_model_training import (
    LanguageModelTrainingStep,
)


def main() -> None:
    tokens = [
        "the",
        "cat",
        "drinks",
        "milk",
        "<eos>",
    ]

    model = SimpleContextLanguageModel(
        vocabulary_size=len(tokens),
        context_size=3,
        model_dimension=16,
        seed=42,
    )

    step = LanguageModelTrainingStep(
        learning_rate=0.05,
    )

    inputs = [0, 1, 2]
    targets = [1, 2, 3]

    first = step.run(
        model=model,
        token_ids=inputs,
        targets=targets,
    )

    result = first

    for _ in range(300):
        result = step.run(
            model=model,
            token_ids=inputs,
            targets=targets,
        )

    print("HowLLMsWork")
    print("============")
    print()
    print(
        "Model: Simple Context Language Model"
    )
    print()
    print(
        f"Initial Loss:       {first.loss:.6f}"
    )
    print(
        f"Final Loss:         {result.loss:.6f}"
    )
    print(
        f"Initial Perplexity: {first.perplexity:.6f}"
    )
    print(
        f"Final Perplexity:   {result.perplexity:.6f}"
    )


if __name__ == "__main__":
    main()
