from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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
                "-m",
                "src.experiments.transformer_session",
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
        head_focuses: list[int],
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
                "head_focuses": head_focuses,
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
