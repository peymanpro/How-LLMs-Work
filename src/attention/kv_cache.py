from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class KVCacheState:
    keys: np.ndarray
    values: np.ndarray


class KVCache:
    def __init__(
        self,
        key_dimension: int,
        value_dimension: int,
    ) -> None:
        if key_dimension <= 0:
            raise ValueError(
                "key_dimension must be positive."
            )

        if value_dimension <= 0:
            raise ValueError(
                "value_dimension must be positive."
            )

        self._key_dimension = key_dimension
        self._value_dimension = value_dimension

        self._keys = np.empty(
            (
                0,
                key_dimension,
            ),
            dtype=np.float64,
        )

        self._values = np.empty(
            (
                0,
                value_dimension,
            ),
            dtype=np.float64,
        )

    @property
    def key_dimension(self) -> int:
        return self._key_dimension

    @property
    def value_dimension(self) -> int:
        return self._value_dimension

    @property
    def sequence_length(self) -> int:
        return self._keys.shape[0]

    @property
    def state(self) -> KVCacheState:
        return KVCacheState(
            keys=self._keys.copy(),
            values=self._values.copy(),
        )

    def append(
        self,
        keys: np.ndarray,
        values: np.ndarray,
    ) -> None:
        new_keys = np.asarray(
            keys,
            dtype=np.float64,
        )

        new_values = np.asarray(
            values,
            dtype=np.float64,
        )

        if new_keys.ndim != 2:
            raise ValueError(
                "Keys must be a two-dimensional matrix."
            )

        if new_values.ndim != 2:
            raise ValueError(
                "Values must be a two-dimensional matrix."
            )

        if new_keys.shape[0] != new_values.shape[0]:
            raise ValueError(
                "Keys and values must contain the same "
                "number of positions."
            )

        if new_keys.shape[1] != self._key_dimension:
            raise ValueError(
                "Key dimension does not match cache."
            )

        if new_values.shape[1] != self._value_dimension:
            raise ValueError(
                "Value dimension does not match cache."
            )

        self._keys = np.concatenate(
            (
                self._keys,
                new_keys,
            ),
            axis=0,
        )

        self._values = np.concatenate(
            (
                self._values,
                new_values,
            ),
            axis=0,
        )

    def truncate_left(
        self,
        count: int,
    ) -> None:
        if count < 0:
            raise ValueError(
                "Truncation count cannot be negative."
            )

        if count > self.sequence_length:
            raise ValueError(
                "Truncation count cannot exceed cache length."
            )

        if count == 0:
            return

        self._keys = self._keys[count:].copy()
        self._values = self._values[count:].copy()

    def clear(
        self,
    ) -> None:
        self._keys = np.empty(
            (
                0,
                self._key_dimension,
            ),
            dtype=np.float64,
        )

        self._values = np.empty(
            (
                0,
                self._value_dimension,
            ),
            dtype=np.float64,
        )

    def get(
        self,
    ) -> KVCacheState:
        return self.state
