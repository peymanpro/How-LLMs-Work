from __future__ import annotations

import numpy as np

from src.attention.qkv_projection import (
    TrainableQKVProjection,
)


def main() -> None:
    projection = TrainableQKVProjection(
        input_dimension=4,
        attention_dimension=3,
        seed=42,
    )

    inputs = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    forward = projection.forward(
        inputs
    )

    gradients = projection.backward(
        forward_result=forward,
        query_gradient=np.ones_like(
            forward.queries
        ),
        key_gradient=np.ones_like(
            forward.keys
        ),
        value_gradient=np.ones_like(
            forward.values
        ),
    )

    print("HowLLMsWork")
    print("============")
    print()
    print(
        f"Input shape:   {inputs.shape}"
    )
    print(
        f"Q shape:       {forward.queries.shape}"
    )
    print(
        f"K shape:       {forward.keys.shape}"
    )
    print(
        f"V shape:       {forward.values.shape}"
    )
    print()
    print(
        "Q sample:"
    )
    print(
        np.array2string(
            forward.queries,
            precision=4,
        )
    )
    print()
    print(
        "Gradient shapes:"
    )
    print(
        f"  dW_Q: {gradients.weights_q.shape}"
    )
    print(
        f"  dW_K: {gradients.weights_k.shape}"
    )
    print(
        f"  dW_V: {gradients.weights_v.shape}"
    )


if __name__ == "__main__":
    main()
