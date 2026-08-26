import numpy as np
import pytest

from src.llm.transformer_language_model import (
    TransformerLanguageModel,
)


class FakeTransformerBackbone:
    @property
    def model_dimension(self) -> int:
        return 4

    @property
    def context_size(self) -> int:
        return 3

    def forward(
        self,
        token_ids: list[int],
    ) -> np.ndarray:
        return np.asarray(
            [
                [
                    float(token_id),
                    1.0,
                    0.5,
                    -1.0,
                ]
                for token_id in token_ids
            ],
            dtype=np.float64,
        )


def create_model() -> TransformerLanguageModel:
    return TransformerLanguageModel(
        backbone=FakeTransformerBackbone(),
        vocabulary_size=6,
        seed=42,
    )


def test_transformer_language_model_should_return_expected_shape() -> None:
    model = create_model()

    logits = model.logits(
        [0, 1, 2]
    )

    assert logits.shape == (
        3,
        6,
    )


def test_transformer_language_model_should_expose_backbone_context_size() -> None:
    model = create_model()

    assert model.context_size == 3
    assert model.model_dimension == 4
    assert model.vocabulary_size == 6


def test_transformer_language_model_should_reject_wrong_backbone_shape() -> None:
    class InvalidBackbone:
        @property
        def model_dimension(self) -> int:
            return 4

        @property
        def context_size(self) -> int:
            return 3

        def forward(
            self,
            token_ids: list[int],
        ) -> np.ndarray:
            return np.zeros(
                (
                    len(token_ids),
                    5,
                ),
                dtype=np.float64,
            )

    model = TransformerLanguageModel(
        backbone=InvalidBackbone(),
        vocabulary_size=6,
    )

    with pytest.raises(ValueError):
        model.logits(
            [0, 1, 2]
        )

from src.llm.language_model import (
    LanguageModel,
)


def test_transformer_language_model_should_match_language_model_contract() -> None:
    model: LanguageModel = create_model()

    logits = model.logits(
        [0, 1, 2]
    )

    assert logits.shape == (
        3,
        6,
    )

from src.training.batch import (
    TensorBatchBuilder,
)
from src.training.dataset import (
    BatchBuilder,
    LanguageModelDataset,
)
from src.training.model_evaluation import (
    BatchLanguageModelEvaluator,
)


def test_transformer_language_model_should_work_with_evaluation_pipeline() -> None:
    model = create_model()

    dataset = LanguageModelDataset(
        token_ids=[
            0,
            1,
            2,
            3,
        ],
        context_size=3,
    )

    batch = BatchBuilder().create_batches(
        dataset,
        batch_size=1,
    )[0]

    tensor_batch = TensorBatchBuilder().build(
        batch
    )

    result = BatchLanguageModelEvaluator().evaluate_tensor_batch(
        model,
        tensor_batch.inputs,
        tensor_batch.targets,
    )

    assert result.examples == 1
    assert np.isfinite(result.loss)
    assert np.isfinite(result.perplexity)
