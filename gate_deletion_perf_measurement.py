#%%
from timeit import default_timer as timer
from bqskit import Circuit
from bqskit.compiler import Compiler, CompilationTask
from bqskit.ir.gates.constant.cx import CXGate
from bqskit.ir.opt.cost.functions.cost.hilbertschmidt import HilbertSchmidtCostGenerator
from bqskit.ir.opt.instantiaters import QFactor_jax_batched_jit
from bqskit.ir.opt.minimizers.lbfgs import LBFGSMinimizer
from bqskit.passes import *



import jax
from bqskit.passes.search.generators.twounitarylayergen import TwoUnitaryLayerGen, TwoUnitaryLayerGenCeres

jax.config.update('jax_enable_x64', True)
#%%
from bqskit import enable_logging, enable_dashboard


# enable_logging(True)
import params

run_params = params.get_params()
print(run_params)

file_path = run_params.input_qasm
file_name = file_path.split("/")[-1]

seed = run_params.seed

print_amount_of_nodes = run_params.print_amount_of_nodes
amount_of_workers = run_params.amount_of_workers
amount_gpus_per_node = run_params.amount_gpus_per_node

use_detached = run_params.use_detached
detached_server_ip = run_params.detached_server_ip
detached_server_port = run_params.detached_server_port


num_multistarts = run_params.multistarts
instantiator = run_params.instantiator
max_iters = run_params.max_iters
min_iters = run_params.min_iters
diff_tol_r = run_params.diff_tol_r
diff_tol_a = run_params.diff_tol_a
dist_tol = run_params.dist_tol

diff_tol_step_r = run_params.diff_tol_step_r
diff_tol_step = run_params.diff_tol_step
beta = run_params.beta

partition_size = run_params.partitions_size
perform_while = run_params.perform_while

print(f"Will compile {file_path}")

batched_instantiation = QFactor_jax_batched_jit(diff_tol_r=diff_tol_r, diff_tol_a=diff_tol_a, min_iters=min_iters, max_iters=max_iters, dist_tol=dist_tol, diff_tol_step_r=diff_tol_step_r, diff_tol_step = diff_tol_step, beta=beta)

instantiator_operated_on_u3s = False

if 'QFACTOR-RUST' in instantiator:

    instantiate_options={
                'method': 'QFactor',
                'diff_tol_r':diff_tol_r,
                'diff_tol_a':diff_tol_a,
                'min_iters':min_iters,
                'dist_tol':dist_tol,  
                'max_iters': max_iters,
                'multistarts': num_multistarts,
                'seed': seed,
            }
elif instantiator == 'QFACTOR-JAX':
    
        instantiate_options={
                    'method': batched_instantiation,
                    'multistarts': num_multistarts,
                    'seed': seed,
                }
elif  'LBFGS' in instantiator:
        instantiator_operated_on_u3s = True
        instantiate_options={
            'method':'minimization',
            'minimizer':LBFGSMinimizer(),
            'multistarts': num_multistarts,
            'cost_fn_gen': HilbertSchmidtCostGenerator(),
            'seed': seed}
else:
    instantiator_operated_on_u3s = True
    instantiate_options={
                    'multistarts': num_multistarts,
                    'seed': seed,
                }

instantiate_options['_P'] = "_P" in instantiator

in_circuit = Circuit.from_file(file_path)

operations_to_perfrom_on_block = [
                    ScanningGateRemovalPass(instantiate_options=instantiate_options),  
                ]

if perform_while:
    operations_to_perfrom_on_block = [
        WhileLoopPass(
            GateCountPredicate(CXGate()),
            operations_to_perfrom_on_block),
        ]     

passes =         [
        # Convert U3's to VU
        FromU3ToVariablePass(),

        QuickPartitioner(partition_size),
        ForEachBlockPass(operations_to_perfrom_on_block),
        UnfoldPass(),
        
        # Convert back to u3 the VU
        ToU3Pass()
        ]


if instantiator_operated_on_u3s:
    passes = passes[1:-1] # no need to convert to variable and then back to U3s

task = CompilationTask(in_circuit.copy(), passes)

if not use_detached:
    compiler =  Compiler(num_workers=amount_of_workers)
else:
    compiler = Compiler(detached_server_ip, detached_server_port)

print(f"Starting {instantiator}")
start = timer()

out_circuit = compiler.compile(task)

end = timer()
run_time = end - start
print(
    f"Partitioning + Synthesis took {run_time}"
    f"seconds using the { instantiator } instantiation method."
)
print(f"Circuit finished with gates: {out_circuit.gate_counts}.")
final_gates_count_by_qudit_number = {g.num_qudits:v for g,v in out_circuit.gate_counts.items()}
print(f"True,{instantiator},{file_name},{num_multistarts},{partition_size},{in_circuit.num_qudits},{run_time},{final_gates_count_by_qudit_number[1]},{final_gates_count_by_qudit_number[2]},{print_amount_of_nodes},{amount_of_workers},{amount_gpus_per_node}")

compiler.close()
