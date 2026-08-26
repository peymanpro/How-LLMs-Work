import pytest

from src.tokenization.tokenizer import WordTokenizer
from src.tokenization.vocabulary import Vocabulary


def create_tokenizer() -> WordTokenizer:
    vocabulary = Vocabulary(
        [
            "the",
            "cat",
            "drinks",
            "milk",
            ".",
        ]
    )

    return WordTokenizer(vocabulary)


def test_vocabulary_should_assign_stable_token_ids() -> None:
    vocabulary = Vocabulary(
        [
            "the",
            "cat",
            "milk",
        ]
    )

    assert vocabulary.size == 3
    assert vocabulary.encode_token("the") == 0
    assert vocabulary.encode_token("cat") == 1
    assert vocabulary.decode_id(2) == "milk"


def test_tokenizer_should_split_words_and_punctuation() -> None:
    tokenizer = create_tokenizer()

    assert tokenizer.tokenize(
        "The cat drinks milk."
    ) == [
        "the",
        "cat",
        "drinks",
        "milk",
        ".",
    ]


def test_tokenizer_should_encode_text() -> None:
    tokenizer = create_tokenizer()

    assert tokenizer.encode(
        "the cat drinks milk."
    ) == [0, 1, 2, 3, 4]


def test_tokenizer_should_decode_token_ids() -> None:
    tokenizer = create_tokenizer()

    assert tokenizer.decode(
        [0, 1, 2, 3, 4]
    ) == "the cat drinks milk ."


def test_tokenizer_should_reject_empty_text() -> None:
    tokenizer = create_tokenizer()

    with pytest.raises(ValueError):
        tokenizer.tokenize("   ")


def test_vocabulary_should_reject_duplicate_tokens() -> None:
    with pytest.raises(ValueError):
        Vocabulary(
            [
                "the",
                "cat",
                "the",
            ]
        )


def test_vocabulary_should_reject_invalid_token_id() -> None:
    vocabulary = Vocabulary(
        [
            "the",
            "cat",
        ]
    )

    with pytest.raises(IndexError):
        vocabulary.decode_id(2)



def test_tokenizer_should_encode_unknown_token_as_unk() -> None:
    vocabulary = Vocabulary(
        [
            "<unk>",
            "the",
            "cat",
        ]
    )

    tokenizer = WordTokenizer(
        vocabulary
    )

    assert tokenizer.encode(
        "the dog"
    ) == [1, 0]


def test_vocabulary_should_expose_unknown_token_id() -> None:
    vocabulary = Vocabulary(
        [
            "<unk>",
            "the",
        ]
    )

    assert vocabulary.unknown_token_id == 0


def test_vocabulary_should_reject_unknown_token_without_unk() -> None:
    vocabulary = Vocabulary(
        [
            "the",
            "cat",
        ]
    )

    with pytest.raises(ValueError):
        vocabulary.encode_token("dog")

