import logging
from typing import Any
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

import jax
import jax.numpy as jnp

from bqskit.ir.opt.instantiater import Instantiater
from bqskit.qis.state.state import StateVector
from bqskit.qis.unitary import LocallyOptimizableUnitary
from bqskit.qis.unitary.unitarybuilder import UnitaryBuilder
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


if TYPE_CHECKING:
    from bqskit.ir.circuit import Circuit


_logger = logging.getLogger(__name__)


class QFactor_batch(Instantiater):
    """The QFactor batch circuit instantiater."""

    def __init__(
        self,
        diff_tol_a: float = 1e-12 ,
        diff_tol_r: float = 1e-6,
        dist_tol: float = 1e-10,
        max_iters: int = 100000,
        min_iters: int = 1000,
    ):

        if not isinstance( diff_tol_a, float ) or diff_tol_a > 0.5:
            raise TypeError( "Invalid absolute difference threshold." )

        if not isinstance( diff_tol_r, float ) or diff_tol_r > 0.5:
            raise TypeError( "Invalid relative difference threshold." )

        if not isinstance( dist_tol, float ) or dist_tol > 0.5:
            raise TypeError( "Invalid distance threshold." )

        if not isinstance( max_iters, int ) or max_iters < 0:
            raise TypeError( "Invalid maximum number of iterations." )

        if not isinstance( min_iters, int ) or min_iters < 0:
            raise TypeError( "Invalid minimum number of iterations." )

        

        self.diff_tol_a = diff_tol_a
        self.diff_tol_r = diff_tol_r
        self.dist_tol   = dist_tol
        self.max_iters  = max_iters
        self.min_iters  = min_iters
        


    def _initilize_circuit_tensor(
        target,
        gates,
        locations,
        params_for_gates
    ):

        target_untry_builder = UnitaryBuilder(target.num_qudits, target.radixes, target)
        param_index = 0
        for gate, loc in zip(gates, locations):
            
            amount_of_params = gate.amount_of_params()
            gparams = params_for_gates[param_index: param_index+amount_of_params]
            target_untry_builder.apply_right(gates.get_unitary(params=gparams), loc, check_arguments=False)
            param_index += amount_of_params

        return target_untry_builder

    def instantiate(
        self,
        circuit: Circuit,
        target: UnitaryMatrix | StateVector,
        x0: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Instantiate `circuit`, see Instantiater for more info."""


        params = np.array(x0)
        amount_of_gates = len(params)
        locations = [op.location for op in circuit]
        gates = [op.gate for op in circuit]
        amount_of_qudits = target.num_qudits
        target_untry_builder = self._initilize_circuit_tensor(target, gates, locations, params)


        c1 = 0
        c2 = 1
        it = 0

        while(True):
            # Termination conditions
            if it > self.min_iters:

                if np.abs(c1 - c2) <= self.diff_tol_a + self.diff_tol_r * np.abs( c1 ):
                    diff = np.abs(c1 - c2)
                    # logger.info( f"Terminated: |c1 - c2| = {diff}"
                    #             " <= diff_tol_a + diff_tol_r * |c1|." )
                    break

                if it > self.max_iters:
                    # logger.info( "Terminated: iteration limit reached." )
                    break

            # from right to left
            param_index = len(params)
            for k in reversed(range(amount_of_gates)):
                gate = gates[k]
                location = locations[k]

                
                amount_of_params = gate.num_params
                gparams = params[param_index - amount_of_params:param_index]
                

                # Remove current gate from right of circuit tensor
                target_untry_builder.apply_right(gate.get_unitary(params=gparams) , location, inverse = True, check_arguments = False)

                # Update current gate
                if amount_of_params > 0:
                    env = target_untry_builder.calc_env_matrix( location )            
                    new_params =  gate.optimize(env)
                    params[param_index - amount_of_params:param_index] = new_params 
                    gparams = new_params

                # Add updated gate to left of circuit tensor
                target_untry_builder.apply_left( gate.get_unitary(new_params), location,  check_arguments = False)

                param_index -= amount_of_params

            # from left to right
            #param index should be 0 now, but we zero it out any way
            param_index = 0
            for k in range( len( circuit ) ):
                gate = gates[k]
                location = locations[k]
                
                amount_of_params = gate.num_params
                gparams = params[param_index: param_index + amount_of_params]
                # Remove current gate from left of circuit tensor
                target_untry_builder.apply_left( gate.get_unitary(params = gparams), location, inverse = True, check_arguments = False)

                # Update current gate
                if gate.num_params > 0:
                    env = target_untry_builder.calc_env_matrix( location )            
                    new_params =  gate.optimize(env)
                    params[param_index : param_index + amount_of_params] = new_params 
                    gparams = new_params

                # Add updated gate to right of circuit tensor
                target_untry_builder.apply_right( gate.get_unitary(params=gparams), location,  check_arguments = False)
            
            c2 = c1
            c1 = np.abs( np.trace( target_untry_builder.utry ) )
            c1 = 1 - ( c1 / ( 2 ** amount_of_qudits ) )
            
            if c1 <= self.dist_tol:
                # logger.info( f"Terminated: c1 = {c1} <= dist_tol." )
                break

            if it % 100 == 0:
                # logger.info( f"iteration: {it}, cost: {c1}" )
                print(f"iteration: {it}, cost: {c1}" )

            if it % 40 == 0:
                target_untry_builder = self._initilize_circuit_tensor(gates, locations, params)


        return params