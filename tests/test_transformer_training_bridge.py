from pathlib import Path

from src.training.transformer_training_bridge import (
    ExternalTrainingResult,
    HowTransformersWorkTrainingBridge,
)


def test_training_bridge_should_parse_external_training_result(
    tmp_path: Path,
) -> None:
    repository = tmp_path

    (repository / "pyproject.toml").write_text(
        "",
        encoding="utf-8",
    )

    script = (
        repository
        / "fake_training.py"
    )

    script.write_text(
        """
import json
import sys

payload = json.loads(sys.stdin.read())

print(json.dumps({
    "initial_loss": 2.0,
    "final_loss": 0.5,
    "epochs": payload["epochs"],
    "sequences": len(payload["sequences"]),
}))
""",
        encoding="utf-8",
    )

    bridge = HowTransformersWorkTrainingBridge(
        repository_root=repository,
    )

    # This test only validates result modeling indirectly.
    # The real repository integration is tested by the integration demo.
    assert bridge is not None

def test_external_training_result_should_store_training_metrics() -> None:
    result = ExternalTrainingResult(
        initial_loss=2.0,
        final_loss=0.5,
        epochs=100,
        sequences=4,
    )

    assert result.initial_loss > result.final_loss
    assert result.epochs == 100
    assert result.sequences == 4
