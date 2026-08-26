from __future__ import annotations

from src.tokenization.tokenizer import WordTokenizer
from src.tokenization.vocabulary import Vocabulary


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

    tokenizer = WordTokenizer(vocabulary)

    text = "The cat drinks milk."

    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.encode(text)

    print("HowLLMsWork")
    print("============")
    print()
    print("Input:")
    print(text)
    print()
    print("Tokens:")
    print(tokens)
    print()
    print("Token IDs:")
    print(token_ids)
    print()
    print("Vocabulary size:")
    print(vocabulary.size)


if __name__ == "__main__":
    main()
