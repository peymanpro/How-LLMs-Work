from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from src.llm.how_transformers_work_adapter import (
    HowTransformersWorkBackboneAdapter,
)
from src.llm.transformer_language_model import (
    TransformerLanguageModel,
)

HOW_TRANSFORMERS_WORK_ROOT = (
    Path(__file__).resolve().parents[3]
    / "HowTransformersWork"
)

BRIDGE = (
    HOW_TRANSFORMERS_WORK_ROOT
    / "src"
    / "experiments"
    / "export_decoder_hidden.py"
)


class ExternalTransformerProxy:
    def __init__(
        self,
        root: Path,
        bridge: Path,
        model_dimension: int,
    ) -> None:
        self._root = root
        self._bridge = bridge
        self._model_dimension = model_dimension

    def forward(
        self,
        token_ids: list[int],
    ) -> object:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.experiments.export_decoder_hidden",
                *(
                    str(token_id)
                    for token_id in token_ids
                ),
            ],
            cwd=self._root,
            capture_output=True,
            text=True,
            check=True,
        )

        payload = json.loads(
            completed.stdout
        )

        data = np.asarray(
            payload["decoder_output"],
            dtype=np.float64,
        )

        class MatrixLike:
            def __init__(
                self,
                values: np.ndarray,
            ) -> None:
                self.data = values

        class Result:
            def __init__(
                self,
                values: np.ndarray,
            ) -> None:
                self.decoder_output = MatrixLike(
                    values
                )

        return Result(data)


def main() -> None:
    if not HOW_TRANSFORMERS_WORK_ROOT.exists():
        raise RuntimeError(
            "HowTransformersWork repository was not found at "
            f"{HOW_TRANSFORMERS_WORK_ROOT}"
        )

    if not BRIDGE.exists():
        raise RuntimeError(
            "HowTransformersWork bridge was not found at "
            f"{BRIDGE}"
        )

    vocabulary_size = 5
    context_size = 4
    model_dimension = 8

    external_model = ExternalTransformerProxy(
        root=HOW_TRANSFORMERS_WORK_ROOT,
        bridge=BRIDGE,
        model_dimension=model_dimension,
    )

    backbone = HowTransformersWorkBackboneAdapter(
        transformer=external_model,
        context_size=context_size,
        model_dimension=model_dimension,
    )

    model = TransformerLanguageModel(
        backbone=backbone,
        vocabulary_size=vocabulary_size,
        seed=42,
    )

    token_ids = [
        0,
        1,
        2,
        3,
    ]

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
        "Backbone: HowTransformersWork"
    )
    print()
    print(
        f"Hidden shape: {hidden.shape}"
    )
    print(
        f"Logits shape: {logits.shape}"
    )


if __name__ == "__main__":
    main()

