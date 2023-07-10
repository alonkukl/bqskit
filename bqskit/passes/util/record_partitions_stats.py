"""This module implements the RecordStatsPass class."""
from __future__ import annotations

import logging
from typing import Any

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates.circuitgate import CircuitGate
from bqskit.ir.operation import Operation

_logger = logging.getLogger(__name__)


class RecordPartitionsStatsPass(BasePass):
    """
    The RecordStatsPass class.

    The RecordStatsPass stores stats about the circuit.
    """

    key = 'RecordPartitionsStatsPass_stats_list'

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        stats: dict[str, Any] = {}
        stats['cycles'] = circuit.num_cycles
        stats['num_ops'] = circuit.num_operations
        stats['cgraph'] = circuit.coupling_graph
        stats['depth'] = circuit.depth
        stats['gate_counts'] = {
            gate: circuit.count(gate)
            for gate in circuit.gate_set
        }


        blocks: list[tuple[int, Operation]] = []
        for cycle, op in circuit.operations_with_cycles():
            blocks.append((cycle, op))

        # No blocks, no work
        if len(blocks) == 0:
            data[self.key].append([])
            return
        
        subcircuits: list[Circuit] = []

        for i, (cycle, op) in enumerate(blocks):

            # Form Subcircuit
            if isinstance(op.gate, CircuitGate):
                subcircuit = op.gate._circuit.copy()
                subcircuit.set_params(op.params)
            else:
                subcircuit = Circuit.from_operation(op)

            subcircuits.append(subcircuit)

        
        qubit_histogram = {}
        gates_histogram = {}

        for circ in subcircuits:
            amount_of_qubits = circ.num_qudits
            qubit_histogram[amount_of_qubits] = 1 + qubit_histogram.get(amount_of_qubits, 0)
            for gate in circ.gate_set:
                d = gates_histogram.get(gate.name, {})
                
                count = circ.count(gate)
                d[count] = 1 + d.get(count, 0)

                gates_histogram[gate.name] = d


        gates_hist = [(g, get_storted_dict(h)) for g,h in gates_histogram.items()]
        _logger.info(' qubit historgram: ' + get_storted_dict(qubit_histogram) + ' total ' + str(sum(v for v in qubit_histogram.values())))
        _logger.info(f' Gates histogram {gates_hist}')
        # _logger.info(f' {circuit.num_qudits} qubits')
        # for gate in circuit.gate_set:
        #     _logger.info(f' {circuit.count(gate)} {gate.name} Count')


def get_storted_dict(d:dict[int, int])->str:
    keys = sorted(list(d.keys()))
    return str([(k,d[k]) for k in keys])