"""This module implements the SinglePhysicalPredicate class."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from bqskit.passes.control.predicate import PassPredicate

if TYPE_CHECKING:
    from bqskit.compiler.passdata import PassData
    from bqskit.ir.circuit import Circuit

_logger = logging.getLogger(__name__)


class SinglePhysicalPredicate(PassPredicate):
    """
    The SinglePhysicalPredicate class.

    The SinglePhysicalPredicate returns true if circuit's single-qudit gates are
    in the native gate set.
    """

    def get_truth_value(self, circuit: Circuit, data: PassData) -> bool:
        """Call this predicate, see :class:`PassPredicate` for more info."""
        model = data.model
        for gate in circuit.gate_set:
            if gate.num_qudits > 1:
                continue
            if gate not in model.gate_set:
                _logger.debug(f'{gate} not in native set: {model.gate_set}.')
                return False

        return True
