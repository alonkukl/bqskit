"""This package defines passes and objects that control pass execution flow."""
from __future__ import annotations

from bqskit.compiler.passes.control.dowhileloop import DoWhileLoopPass
from bqskit.compiler.passes.control.foreach import ForEachBlockPass
from bqskit.compiler.passes.control.ifthenelse import IfThenElsePass
from bqskit.compiler.passes.control.predicate import PassPredicate
from bqskit.compiler.passes.control.predicates.change import ChangePredicate
from bqskit.compiler.passes.control.predicates.count import GateCountPredicate
from bqskit.compiler.passes.control.predicates.notpredicate import NotPredicate
from bqskit.compiler.passes.control.whileloop import WhileLoopPass

__all__ = [
    'DoWhileLoopPass',
    'ForEachBlockPass',
    'IfThenElsePass',
    'PassPredicate',
    'ChangePredicate',
    'GateCountPredicate',
    'NotPredicate',
    'WhileLoopPass',
]
