from __future__ import annotations

from pathlib import Path

from src.training.transformer_training_bridge import (
    HowTransformersWorkTrainingBridge,
)


def main() -> None:
    repository = (
        Path(__file__).resolve().parents[3]
        / "How-Transformers-Work"
    )

    if not repository.exists():
        raise RuntimeError(
            "How-Transformers-Work repository was not found at "
            f"{repository}"
        )

    bridge = HowTransformersWorkTrainingBridge(
        repository_root=repository,
    )

    result = bridge.train(
        vocabulary_size=6,
        model_dimension=8,
        head_dimension=2,
        feed_forward_dimension=16,
        maximum_sequence_length=8,
        learning_rate=0.05,
        epochs=100,
        sequences=[
            [0, 1, 2, 3],
        ],
        targets=[
            [1, 2, 3, 4],
        ],
    )

    print("HowLLMsWork")
    print("============")
    print()
    print(
        "Backbone: How-Transformers-Work"
    )
    print(
        f"Initial Loss: {result.initial_loss:.6f}"
    )
    print(
        f"Final Loss:   {result.final_loss:.6f}"
    )
    print(
        f"Reduction:    "
        f"{result.initial_loss - result.final_loss:.6f}"
    )


if __name__ == "__main__":
    main()
