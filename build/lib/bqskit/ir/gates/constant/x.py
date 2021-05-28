"""This module implements the XGate."""
from __future__ import annotations

from bqskit.ir.gates.constantgate import ConstantGate
from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


class XGate(ConstantGate, QubitGate):
    """The Pauli X gate."""

    size = 1
    qasm_name = 'x'
    utry = UnitaryMatrix(
        [
            [0, 1],
            [1, 0],
        ],
    )
