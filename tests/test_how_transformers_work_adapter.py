import numpy as np
import pytest

from src.llm.how_transformers_work_adapter import (
    HowTransformersWorkBackboneAdapter,
)


class FakeMatrix:
    def __init__(
        self,
        data: np.ndarray,
    ) -> None:
        self.data = data


class FakeForwardResult:
    def __init__(
        self,
        decoder_output: FakeMatrix,
    ) -> None:
        self.decoder_output = decoder_output


class FakeTransformer:
    def __init__(
        self,
        model_dimension: int = 8,
    ) -> None:
        self._model_dimension = model_dimension

    def forward(
        self,
        token_ids: list[int],
    ) -> FakeForwardResult:
        hidden = np.ones(
            (
                len(token_ids),
                self._model_dimension,
            ),
            dtype=np.float64,
        )

        return FakeForwardResult(
            decoder_output=FakeMatrix(
                hidden
            )
        )


def test_adapter_should_return_decoder_output_as_ndarray() -> None:
    adapter = HowTransformersWorkBackboneAdapter(
        transformer=FakeTransformer(),
        context_size=4,
        model_dimension=8,
    )

    hidden = adapter.forward(
        [0, 1, 2, 3]
    )

    assert hidden.shape == (
        4,
        8,
    )

    assert hidden.dtype == np.float64


def test_adapter_should_expose_backbone_dimensions() -> None:
    adapter = HowTransformersWorkBackboneAdapter(
        transformer=FakeTransformer(),
        context_size=4,
        model_dimension=8,
    )

    assert adapter.context_size == 4
    assert adapter.model_dimension == 8


def test_adapter_should_reject_wrong_context_length() -> None:
    adapter = HowTransformersWorkBackboneAdapter(
        transformer=FakeTransformer(),
        context_size=4,
        model_dimension=8,
    )

    with pytest.raises(ValueError):
        adapter.forward(
            [0, 1, 2]
        )


def test_adapter_should_reject_unexpected_hidden_shape() -> None:
    class InvalidTransformer:
        def forward(
            self,
            token_ids: list[int],
        ) -> FakeForwardResult:
            return FakeForwardResult(
                decoder_output=FakeMatrix(
                    np.zeros(
                        (
                            len(token_ids),
                            7,
                        ),
                        dtype=np.float64,
                    )
                )
            )

    adapter = HowTransformersWorkBackboneAdapter(
        transformer=InvalidTransformer(),
        context_size=4,
        model_dimension=8,
    )

    with pytest.raises(ValueError):
        adapter.forward(
            [0, 1, 2, 3]
        )
