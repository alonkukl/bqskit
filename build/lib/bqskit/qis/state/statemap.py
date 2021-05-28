"""This module implements the StateVectorMap base class."""
from __future__ import annotations

import abc

from bqskit.qis.state.state import StateVector


class StateVectorMap(abc.ABC):
    """The StateVectorMap base class."""

    @abc.abstractmethod
    def get_statevector(self, in_state: StateVector) -> StateVector:
        """Calculate the output state given the input state."""
