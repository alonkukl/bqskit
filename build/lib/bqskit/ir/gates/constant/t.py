"""This module implements the TGate."""
from __future__ import annotations

import numpy as np

from bqskit.ir.gates.constantgate import ConstantGate
from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


class TGate(ConstantGate, QubitGate):
    """The T gate."""

    size = 1
    qasm_name = 't'
    utry = UnitaryMatrix(
        [
            [1, 0],
            [0, np.exp(1j * np.pi / 4)],
        ],
    )
