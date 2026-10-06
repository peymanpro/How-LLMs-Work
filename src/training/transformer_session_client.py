from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

$1 = r'''
import json
import sys

from src.transformer.language_model import TinyTransformerLanguageModel
from src.transformer.training import TransformerTrainer


model = None
trainer = None


def respond(payload):
    print(json.dumps(payload), flush=True)


for line in sys.stdin:
    try:
        request = json.loads(line)
        command = request["command"]

        if command == "initialize":
            model = TinyTransformerLanguageModel(
                vocabulary_size=int(request["vocabulary_size"]),
                model_dimension=int(request["model_dimension"]),
                head_dimension=int(request["head_dimension"]),
                feed_forward_dimension=int(
                    request["feed_forward_dimension"]
                ),
                maximum_sequence_length=int(
                    request["maximum_sequence_length"]
                ),
                seed=int(request.get("seed", 42)),
            )
            trainer = TransformerTrainer(
                model=model,
                learning_rate=float(request["learning_rate"]),
            )
            respond({"ok": True})
            continue

        if model is None or trainer is None:
            raise RuntimeError(
                "Transformer session has not been initialized."
            )

        if command == "forward":
            token_ids = [int(value) for value in request["token_ids"]]
            result = model.forward(token_ids)
            respond(
                {
                    "ok": True,
                    "logits": result.logits.values.data.tolist(),
                    "decoder_output": (
                        result.decoder_output.data.tolist()
                    ),
                }
            )
            continue

        if command == "evaluate":
            result = trainer.evaluate(
                sequences=request["sequences"],
                targets=request["targets"],
            )
            respond(
                {
                    "ok": True,
                    "loss": float(result),
                }
            )
            continue

        if command == "train":
            history = trainer.train(
                sequences=[request["token_ids"]],
                targets=[request["targets"]],
                epochs=1,
            )
            respond(
                {
                    "ok": True,
                    "loss": float(history[-1].average_loss),
                }
            )
            continue

        if command == "train_many":
            history = trainer.train(
                sequences=request["sequences"],
                targets=request["targets"],
                epochs=int(request["epochs"]),
            )
            respond(
                {
                    "ok": True,
                    "loss": float(history[-1].average_loss),
                }
            )
            continue

        if command == "shutdown":
            respond({"ok": True})
            break

        raise ValueError(f"Unknown command: {command}")

    except Exception as error:
        respond(
            {
                "ok": False,
                "error": str(error),
            }
        )
'''


@dataclass(frozen=True)
class ForwardResult:
    logits: list[list[float]]
    decoder_output: list[list[float]]


@dataclass(frozen=True)
class TrainResult:
    loss: float


class TransformerSessionClient:
    def __init__(
        self,
        repository_root: Path,
    ) -> None:
        self._process = subprocess.Popen(
            [
                sys.executable,
                "-c",
                _SESSION_SERVER,
            ],
            cwd=repository_root,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

    def _request(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        if (
            self._process.stdin is None
            or self._process.stdout is None
        ):
            raise RuntimeError(
                "Transformer session streams are unavailable."
            )

        if self._process.poll() is not None:
            stderr = ""

            if self._process.stderr is not None:
                stderr = self._process.stderr.read()

            raise RuntimeError(
                "Transformer session terminated before request. "
                f"stderr: {stderr}"
            )

        try:
            self._process.stdin.write(
                json.dumps(payload)
                + "\n"
            )
            self._process.stdin.flush()

            line = self._process.stdout.readline()

        except OSError as error:
            stderr = ""

            if self._process.stderr is not None:
                stderr = self._process.stderr.read()

            raise RuntimeError(
                "Failed to communicate with transformer session. "
                f"stderr: {stderr}"
            ) from error

        if not line:
            stderr = ""

            if self._process.stderr is not None:
                stderr = self._process.stderr.read()

            raise RuntimeError(
                "Transformer session terminated unexpectedly. "
                f"stderr: {stderr}"
            )

        response = json.loads(line)

        if not response.get("ok", False):
            raise RuntimeError(
                response.get(
                    "error",
                    "Unknown transformer session error.",
                )
            )

        return response

    def initialize(
        self,
        *,
        vocabulary_size: int,
        model_dimension: int,
        head_dimension: int,
        feed_forward_dimension: int,
        maximum_sequence_length: int,
        learning_rate: float,
        seed: int = 42,
    ) -> None:
        self._request(
            {
                "command": "initialize",
                "vocabulary_size": vocabulary_size,
                "model_dimension": model_dimension,
                "head_dimension": head_dimension,
                "feed_forward_dimension": (
                    feed_forward_dimension
                ),
                "maximum_sequence_length": (
                    maximum_sequence_length
                ),
                "learning_rate": learning_rate,
                "seed": seed,
            }
        )

    def train(
        self,
        token_ids: list[int],
        targets: list[int],
    ) -> TrainResult:
        response = self._request(
            {
                "command": "train",
                "token_ids": token_ids,
                "targets": targets,
            }
        )

        return TrainResult(
            loss=float(
                response["loss"]
            )
        )

    def train_many(
        self,
        sequences: list[list[int]],
        targets: list[list[int]],
        epochs: int,
    ) -> TrainResult:
        response = self._request(
            {
                "command": "train_many",
                "sequences": sequences,
                "targets": targets,
                "epochs": epochs,
            }
        )

        return TrainResult(
            loss=float(
                response["loss"]
            )
        )

    def evaluate(
        self,
        sequences: list[list[int]],
        targets: list[list[int]],
    ) -> float:
        response = self._request(
            {
                "command": "evaluate",
                "sequences": sequences,
                "targets": targets,
            }
        )

        return float(response["loss"])

    def forward(
        self,
        token_ids: list[int],
    ) -> ForwardResult:
        response = self._request(
            {
                "command": "forward",
                "token_ids": token_ids,
            }
        )

        return ForwardResult(
            logits=response["logits"],
            decoder_output=response[
                "decoder_output"
            ],
        )

    def predict_next_token(
        self,
        token_ids: list[int],
    ) -> list[list[float]]:
        result = self.forward(
            token_ids
        )

        return result.logits

    def close(self) -> None:
        if self._process.poll() is not None:
            return

        try:
            self._request(
                {
                    "command": "shutdown"
                }
            )
        finally:
            self._process.wait()
