"""This module implements the YGate."""
from __future__ import annotations

from bqskit.ir.gates.constantgate import ConstantGate
from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


class YGate(ConstantGate, QubitGate):
    """The Pauli Y gate."""

    size = 1
    qasm_name = 'y'
    utry = UnitaryMatrix(
        [
            [0, -1j],
            [1j, 0],
        ],
    )
