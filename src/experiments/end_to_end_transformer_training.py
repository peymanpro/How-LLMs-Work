from __future__ import annotations

from pathlib import Path

from src.training.transformer_training_bridge import (
    HowTransformersWorkTrainingBridge,
)


def main() -> None:
    repository = (
        Path(__file__).resolve().parents[3]
        / "HowTransformersWork"
    )

    bridge = HowTransformersWorkTrainingBridge(
        repository_root=repository,
    )

    result = bridge.train(
        vocabulary_size=5,
        model_dimension=8,
        head_dimension=4,
        head_focuses=[0, 1],
        feed_forward_dimension=16,
        maximum_sequence_length=4,
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
        "Backbone: HowTransformersWork"
    )
    print()
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
