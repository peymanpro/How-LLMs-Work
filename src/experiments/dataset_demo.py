from __future__ import annotations

from src.tokenization.tokenizer import WordTokenizer
from src.tokenization.vocabulary import Vocabulary
from src.training.dataset import (
    BatchBuilder,
    LanguageModelDataset,
)


def main() -> None:
    vocabulary = Vocabulary(
        [
            "<unk>",
            "the",
            "cat",
            "drinks",
            "milk",
            "dog",
        ]
    )

    tokenizer = WordTokenizer(
        vocabulary
    )

    text = (
        "the cat drinks milk "
        "the dog drinks milk"
    )

    token_ids = tokenizer.encode(
        text
    )

    dataset = LanguageModelDataset(
        token_ids=token_ids,
        context_size=4,
    )

    batches = BatchBuilder().create_batches(
        dataset,
        batch_size=2,
    )

    print("HowLLMsWork")
    print("============")
    print()
    print("Token IDs:")
    print(token_ids)
    print()
    print("Dataset size:")
    print(dataset.size)
    print()
    print("Examples:")

    for index, example in enumerate(
        dataset.examples()
    ):
        print(
            f"{index}: "
            f"{example.inputs} -> "
            f"{example.targets}"
        )

    print()
    print("Batches:")

    for index, batch in enumerate(batches):
        print(
            f"Batch {index}:"
        )
        print(
            f"  inputs = {batch.inputs}"
        )
        print(
            f"  targets = {batch.targets}"
        )


if __name__ == "__main__":
    main()

