from __future__ import annotations

from pathlib import Path

from src.llm.how_transformers_work_adapter import (
    HowTransformersWorkBackboneAdapter,
)
from src.llm.transformer_language_model import (
    TransformerLanguageModel,
)
from src.training.transformer_session_client import (
    TransformerSessionClient,
)


HOW_TRANSFORMERS_WORK_ROOT = (
    Path(__file__).resolve().parents[3]
    / "How-Transformers-Work"
)


class SessionTransformerProxy:
    def __init__(
        self,
        client: TransformerSessionClient,
    ) -> None:
        self._client = client

    def forward(
        self,
        token_ids: list[int],
    ) -> object:
        result = self._client.forward(
            token_ids
        )

        class MatrixLike:
            def __init__(
                self,
                values: list[list[float]],
            ) -> None:
                self.data = values

        class Result:
            def __init__(
                self,
                values: list[list[float]],
            ) -> None:
                self.decoder_output = MatrixLike(
                    values
                )

        return Result(
            result.decoder_output
        )


def main() -> None:
    if not HOW_TRANSFORMERS_WORK_ROOT.exists():
        raise RuntimeError(
            "How-Transformers-Work repository was not found at "
            f"{HOW_TRANSFORMERS_WORK_ROOT}"
        )

    client = TransformerSessionClient(
        HOW_TRANSFORMERS_WORK_ROOT
    )

    try:
        client.initialize(
            vocabulary_size=6,
            model_dimension=8,
            head_dimension=2,
            feed_forward_dimension=16,
            maximum_sequence_length=8,
            learning_rate=0.05,
        )

        backbone = HowTransformersWorkBackboneAdapter(
            transformer=SessionTransformerProxy(client),
            context_size=4,
            model_dimension=8,
        )

        model = TransformerLanguageModel(
            backbone=backbone,
            vocabulary_size=6,
            seed=42,
        )

        token_ids = [0, 1, 2, 3]

        hidden = backbone.forward(
            token_ids
        )

        logits = model.logits(
            token_ids
        )

        print("HowLLMsWork")
        print("============")
        print()
        print(
            "Backbone: How-Transformers-Work"
        )
        print(
            f"Hidden shape: {hidden.shape}"
        )
        print(
            f"Logits shape: {logits.shape}"
        )
    finally:
        client.close()


if __name__ == "__main__":
    main()
