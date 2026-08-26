from __future__ import annotations

import numpy as np

from src.llm.positional_language_model import (
    PositionalContextLanguageModel,
)
from src.llm.simple_language_model import (
    SimpleContextLanguageModel,
)

TOKENS = [
    "the",
    "cat",
    "drinks",
    "milk",
]


def main() -> None:
    simple = SimpleContextLanguageModel(
        vocabulary_size=4,
        context_size=3,
        model_dimension=8,
        seed=42,
    )

    positional = PositionalContextLanguageModel(
        vocabulary_size=4,
        context_size=3,
        model_dimension=8,
        seed=42,
    )

    first = [0, 1, 2]
    reordered = [2, 1, 0]

    simple_a = simple.forward(
        first
    )

    simple_b = simple.forward(
        reordered
    )

    positional_a = positional.forward(
        first
    )

    positional_b = positional.forward(
        reordered
    )

    print("HowLLMsWork")
    print("============")
    print()

    print("Sequence A:")
    print(
        " ".join(
            TOKENS[token_id]
            for token_id in first
        )
    )

    print()

    print("Sequence B:")
    print(
        " ".join(
            TOKENS[token_id]
            for token_id in reordered
        )
    )

    print()

    print(
        "Simple model — hidden states equal:"
    )

    print(
        np.allclose(
            simple_a.hidden,
            simple_b.hidden,
        )
    )

    print()

    print(
        "Positional model — hidden states equal:"
    )

    print(
        np.allclose(
            positional_a.hidden,
            positional_b.hidden,
        )
    )


if __name__ == "__main__":
    main()
