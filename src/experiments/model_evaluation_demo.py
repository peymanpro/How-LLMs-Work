from __future__ import annotations

from src.llm.transition_model import TransitionTableLanguageModel
from src.training.dataset import BatchBuilder, LanguageModelDataset
from src.training.model_evaluation import BatchLanguageModelEvaluator


def main() -> None:
    model = TransitionTableLanguageModel(
        vocabulary_size=5,
        context_size=4,
        transitions={
            0: 1,
            1: 2,
            2: 3,
            3: 4,
        },
    )

    dataset = LanguageModelDataset(
        token_ids=[0, 1, 2, 3, 4],
        context_size=4,
    )

    batches = BatchBuilder().create_batches(
        dataset,
        batch_size=1,
    )

    evaluator = BatchLanguageModelEvaluator()

    print("HowLLMsWork")
    print("============")
    print()

    for index, batch in enumerate(batches, start=1):
        result = evaluator.evaluate_batch(model, batch)

        print(f"Batch {index}:")
        print(f"  Loss:       {result.loss:.6f}")
        print(f"  Perplexity: {result.perplexity:.6f}")
        print()


if __name__ == "__main__":
    main()
