"""This module implements the RZZGate."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.qis.unitary.differentiable import DifferentiableUnitary
from bqskit.qis.unitary.optimizable import LocallyOptimizableUnitary
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


class RZZGate(QubitGate, DifferentiableUnitary, LocallyOptimizableUnitary):
    """A gate representing an arbitrary rotation around the ZZ axis."""

    size = 2
    num_params = 1
    qasm_name = 'rzz'

    def get_unitary(self, params: Sequence[float] = []) -> UnitaryMatrix:
        """Returns the unitary for this gate, see Unitary for more info."""
        self.check_parameters(params)

        pos = np.exp(1j * params[0] / 2)
        neg = np.exp(-1j * params[0] / 2)

        return UnitaryMatrix(
            [
                [neg, 0, 0, 0],
                [0, pos, 0, 0],
                [0, 0, pos, 0],
                [0, 0, 0, neg],
            ],
        )

    def get_grad(self, params: Sequence[float] = []) -> np.ndarray:
        """Returns the gradient for this gate, see Gate for more info."""
        self.check_parameters(params)

        dpos = 1j / 2 * np.exp(1j * params[0] / 2)
        dneg = -1j / 2 * np.exp(-1j * params[0] / 2)

        return np.array(
            [
                [
                    [dneg, 0, 0, 0],
                    [0, dpos, 0, 0],
                    [0, 0, dpos, 0],
                    [0, 0, 0, dneg],
                ],
            ], dtype=np.complex128,
        )

    def optimize(self, env_matrix: np.ndarray) -> list[float]:
        """Returns optimal parameters with respect to an environment matrix."""
        self.check_env_matrix(env_matrix)
        raise NotImplementedError()  # TODO
