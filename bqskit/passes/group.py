from __future__ import annotations

from typing import Sequence

from bqskit.passes.alias import PassAlias
from bqskit.compiler import BasePass
from bqskit.utils.typing import is_sequence

class PassGroup(PassAlias):
    """A pass that is a group of other passes."""

    def __init__(self, passes: BasePass | Sequence[BasePass]) -> None:
        """Group together one or more `passes`."""
        self.passes = passes if is_sequence(passes) else [passes]

        for p in self.passes:
            if not isinstance(p, BasePass):
                raise TypeError(f"Expected a BasePass, got {type(p)}.")
    
    def get_passes(self) -> list[BasePass]:
        """Return the passes to be run, see :class:`PassAlias` for more."""
        return self.passes
