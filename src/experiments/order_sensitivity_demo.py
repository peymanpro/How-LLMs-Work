from __future__ import annotations

import numpy as np

from src.llm.simple_language_model import (
    SimpleContextLanguageModel,
)

TOKENS = [
    "the",
    "cat",
    "drinks",
    "milk",
]


def describe(
    ids: list[int],
) -> str:
    return " ".join(
        TOKENS[token_id]
        for token_id in ids
    )


def main() -> None:
    model = SimpleContextLanguageModel(
        vocabulary_size=len(TOKENS),
        context_size=3,
        model_dimension=8,
        seed=42,
    )

    first = [0, 1, 2]
    reordered = [2, 1, 0]

    first_result = model.forward(
        first
    )

    reordered_result = model.forward(
        reordered
    )

    print("HowLLMsWork")
    print("============")
    print()
    print("Sequence A:")
    print(
        f"  {describe(first)}"
    )
    print()
    print("Sequence B:")
    print(
        f"  {describe(reordered)}"
    )
    print()
    print(
        "Hidden representations equal:"
    )
    print(
        np.allclose(
            first_result.hidden,
            reordered_result.hidden,
        )
    )
    print()
    print(
        "Logits equal:"
    )
    print(
        np.allclose(
            first_result.logits,
            reordered_result.logits,
        )
    )
    print()
    print(
        "Conclusion:"
    )
    print(
        "The mean-based baseline cannot represent token order."
    )


if __name__ == "__main__":
    main()
