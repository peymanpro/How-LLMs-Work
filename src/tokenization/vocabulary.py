from __future__ import annotations


class Vocabulary:
    def __init__(
        self,
        tokens: list[str],
    ) -> None:
        if not tokens:
            raise ValueError(
                "Vocabulary cannot be empty."
            )

        if len(tokens) != len(set(tokens)):
            raise ValueError(
                "Vocabulary tokens must be unique."
            )

        self._tokens = tuple(tokens)

        self._token_to_id = {
            token: index
            for index, token in enumerate(self._tokens)
        }

    @property
    def size(self) -> int:
        return len(self._tokens)

    def encode_token(
        self,
        token: str,
    ) -> int:
        try:
            return self._token_to_id[token]
        except KeyError as error:
            raise KeyError(
                f"Unknown token: {token!r}"
            ) from error

    def decode_id(
        self,
        token_id: int,
    ) -> str:
        if not 0 <= token_id < self.size:
            raise IndexError(
                "Token ID is outside the vocabulary."
            )

        return self._tokens[token_id]

    def tokens(self) -> tuple[str, ...]:
        return self._tokens
