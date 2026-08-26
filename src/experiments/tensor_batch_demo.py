from __future__ import annotations

from src.training.batch import (
    TensorBatchBuilder,
)
from src.training.dataset import (
    BatchBuilder,
    LanguageModelDataset,
)


def main() -> None:
    dataset = LanguageModelDataset(
        token_ids=[
            0,
            1,
            2,
            3,
            4,
            5,
        ],
        context_size=3,
    )

    batch = BatchBuilder().create_batches(
        dataset,
        batch_size=2,
    )[0]

    tensor_batch = TensorBatchBuilder().build(
        batch
    )

    print("HowLLMsWork")
    print("============")
    print()
    print("Inputs:")
    print(tensor_batch.inputs)
    print()
    print("Targets:")
    print(tensor_batch.targets)
    print()
    print(
        f"Input shape:  {tensor_batch.inputs.shape}"
    )
    print(
        f"Target shape: {tensor_batch.targets.shape}"
    )
    print(
        f"Input dtype:  {tensor_batch.inputs.dtype}"
    )


if __name__ == "__main__":
    main()
