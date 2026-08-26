from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CausalTrainingExample:
    inputs: tuple[int, ...]
    targets: tuple[int, ...]


class CausalExampleBuilder:
    def build(
        self,
        token_ids: list[int],
    ) -> CausalTrainingExample:
        if len(token_ids) < 2:
            raise ValueError(
                "At least two tokens are required."
            )

        return CausalTrainingExample(
            inputs=tuple(token_ids[:-1]),
            targets=tuple(token_ids[1:]),
        )
