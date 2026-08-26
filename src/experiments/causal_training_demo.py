from __future__ import annotations

from src.tokenization.tokenizer import WordTokenizer
from src.tokenization.vocabulary import Vocabulary
from src.training.causal_examples import (
    CausalExampleBuilder,
)


def main() -> None:
    vocabulary = Vocabulary(
        [
            "<unk>",
            "the",
            "cat",
            "drinks",
            "milk",
            ".",
        ]
    )

    tokenizer = WordTokenizer(
        vocabulary
    )

    text = "The cat drinks milk."

    token_ids = tokenizer.encode(
        text
    )

    example = CausalExampleBuilder().build(
        token_ids
    )

    print("HowLLMsWork")
    print("============")
    print()
    print("Text:")
    print(text)
    print()
    print("Token IDs:")
    print(token_ids)
    print()
    print("Input IDs:")
    print(example.inputs)
    print()
    print("Target IDs:")
    print(example.targets)
    print()
    print("Learning objective:")
    print(
        "predict the next token from previous tokens"
    )


if __name__ == "__main__":
    main()
