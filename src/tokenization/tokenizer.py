from __future__ import annotations

import re

from src.tokenization.vocabulary import Vocabulary


_TOKEN_PATTERN = re.compile(
    r"\w+|[^\w\s]"
)


class WordTokenizer:
    def __init__(
        self,
        vocabulary: Vocabulary,
    ) -> None:
        self._vocabulary = vocabulary

    def tokenize(
        self,
        text: str,
    ) -> list[str]:
        if not text.strip():
            raise ValueError(
                "Text cannot be empty."
            )

        return _TOKEN_PATTERN.findall(
            text.lower()
        )

    def encode(
        self,
        text: str,
    ) -> list[int]:
        tokens = self.tokenize(text)

        return [
            self._vocabulary.encode_token(token)
            for token in tokens
        ]

    def decode(
        self,
        token_ids: list[int],
    ) -> str:
        tokens = [
            self._vocabulary.decode_id(token_id)
            for token_id in token_ids
        ]

        return " ".join(tokens)
