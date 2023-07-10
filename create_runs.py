import os
import sys
import subprocess 
import time

file_name = 'job.sh'
sleep_time = 0.05
global_i = 0

detached_runtime_header="""#!/bin/bash
#SBATCH --job-name=syn{i}_{cir_name}_{instantiator}_detached_runtime
#SBATCH -A {account}
#SBATCH -C {constraint}
#SBATCH -q regular
#SBATCH -t {time_limit}
#SBATCH -n {nodes}
#SBATCH --mem=0
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task={amount_of_gpus_per_node}
#SBATCH --output=./logs_qce_more_runs/{cir_name}_{partitions_size}p_{instantiator}_detached_runtime_{nodes}_nodes_{num_multistarts}_starts-%j.txt


#sdfsgTCH --output=./logs_qfactor_jax_new_termination_cond/{cir_name}_{partitions_size}p_{instantiator}_detached_runtime_{nodes}_nodes_{num_multistarts}_starts-%j.txt

date
uname -a
module load python
# conda activate my_env
conda activate dev_env

module load nvidia



echo "starting BQSKit managers on all nodes"
srun run_workers_and_managers.sh {amount_of_gpus_per_node} {amount_of_workers_per_gpu} &
managers_pid=$!

filename=$SCRATCH/managers_${{SLURM_JOB_ID}}_started
n={nodes}

while [[ ! -f "$filename" ]]
do
        sleep 0.5
done

while [ "$(cat "$filename" | wc -l)" -lt "$n" ]; do
    sleep 1
done

echo "starting BQSKit server on main node"
bqskit-server $(scontrol show hostnames "$SLURM_JOB_NODELIST" | tr '\\n' ' ') &> $SCRATCH/bqskit_logs/server_${{SLURM_JOB_ID}}.log &
server_pid=$!

uname -a >> $SCRATCH/server_${{SLURM_JOB_ID}}_started

echo "will run {env_vars} python  ./{python_file} --diff_tol_a 0  --use_detached {command_args} "

{env_vars} python  ./{python_file} --diff_tol_a 0 --use_detached {command_args}


echo "Killing the server"
kill -2 $server_pid

sleep 2

"""


attached_runtime_header="""#!/bin/bash
#SBATCH --job-name=syn{i}_{cir_name}_{instantiator}_attached_runtime
#SBATCH -A {account}
#SBATCH -C {constraint}
#SBATCH -q regular
#SBATCH -t {time_limit}
#SBATCH -n {nodes}
#SBATCH --mem=0
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task={amount_of_gpus_per_node}
#SBATCH --output=./logs_qce_more_runs/{cir_name}_{partitions_size}p_{instantiator}_attached_runtime_{nodes}_nodes_{num_multistarts}_starts-%j.txt



#alon --output=./logs_qfactor_jax_new_termination_cond/{cir_name}_{partitions_size}p_{instantiator}_attached_runtime_{nodes}_nodes_{num_multistarts}_starts-%j.txt

date
uname -a
module load python
conda activate my_env
# conda activate dev_env

module load nvidia

echo "starting MPS server on node"
nvidia-cuda-mps-control -d

echo "will run {env_vars} python  ./{python_file}  --diff_tol_a 0 {command_args} "

{env_vars} python  ./{python_file}  --diff_tol_a 0 {command_args}

sleep 1
echo "Trying to stop MPS on all nodes"
echo quit | nvidia-cuda-mps-control

# {amount_of_workers_per_gpu}

"""


def create_and_run_a_job(cir_path, partitions_size, python_file,
        instantiator, num_multistarts, seed, nodes, workers_per_node,
        amount_of_gpus_per_node, use_detached, time_limit):
    
    global global_i

    to_write = open(file_name, 'w')
    global_i += 1
    command_args = f"--input_qasm {cir_path} --multistarts {num_multistarts} --partitions_size {partitions_size} --print_amount_of_nodes {nodes} --instantiator {instantiator}"
    env_vars = ""
    
    if seed is not None:
        command_args += f" --seed {seed}"

    if instantiator == 'QFACTOR-JAX':

        command_args += f' --amount_of_workers {workers_per_node}'
        command_args += f' --amount_gpus_per_node {amount_of_gpus_per_node}'
        if not use_detached:
            env_vars += " XLA_PYTHON_CLIENT_PREALLOCATE=false"
            env_vars += " CUDA_VISIBLE_DEVICES=" + str(list(range(amount_of_gpus_per_node)))[1:-1].replace(" ","")
    else:
        if not use_detached:
            workers_per_node = -1 
        command_args += f' --amount_of_workers {workers_per_node}'
        command_args += f' --amount_gpus_per_node 0'

    if amount_of_gpus_per_node == 0:

        amount_of_workers_per_gpu = workers_per_node
    else:
        amount_of_workers_per_gpu = workers_per_node // amount_of_gpus_per_node


    if use_detached:
        runtime_type='detached'
        header_to_use = detached_runtime_header
    else:
        assert(nodes==1)
        runtime_type='attached'
        header_to_use = attached_runtime_header

    cir_name = cir_path.split("/")[-1].split(".qasm")[0]
    account = 'm4141'
    if 'JAX' in instantiator:
        account+='_g'
        constarint = 'gpu'
    else:
        constarint = 'cpu'
    to_write.write(header_to_use.format(i=global_i,
                        cir_name=cir_name, instantiator=instantiator,
                         partitions_size=partitions_size, account=account,
                        num_multistarts=num_multistarts, time_limit=time_limit,
                        env_vars=env_vars, command_args=command_args,
                        nodes = nodes, amount_of_gpus_per_node=amount_of_gpus_per_node,
                        workers_per_node=workers_per_node, python_file=python_file,
                        runtime_type=runtime_type,constraint=constarint,
                        amount_of_workers_per_gpu=amount_of_workers_per_gpu))

    to_write.close()
    time.sleep(2*sleep_time)

    subprocess.check_output(['sbatch' , file_name])
    time.sleep(sleep_time)



if __name__ == '__main__':  

    # circuits =  [f'{f}.qasm' for f in ['qaoa10_u3', 'qaoa12_u3', 'mul10_u3']]

    # circuits =  [('qaoa5.qasm', 5), ('grover5.qasm', 5), ('adder9.qasm', 9), ('hub4.qasm', 4)]
    # circuits =  [('qaoa5.qasm', 5), ('grover5.qasm', 5), ('adder9.qasm', 9)]
    # circuits =  [('qaoa5.qasm', 5), ('grover5.qasm', 5)]
    # circuits =  [ ('qaoa12_u3.qasm', 12)]
    # circuits =  [ ('adder63_u3.qasm', 63), ('shor26.qasm', 26), ('hub18.qasm', 18)]
    # circuits =  [ ('adder63_u3.qasm', 63)]
    # circuits = [ ('tfim8.qasm', 8), ('tfim16.qasm', 16), ('qpe8.qasm', 8), ('qpe10.qasm', 10), ('qpe12.qasm', 12), ('hhl8.qasm', 8),  ('heisenberg8.qasm', 8),   ('heisenberg7.qasm', 7), ('qae11.qasm', 11), ('qae13.qasm', 13)]
    # circuits = [('qae11.qasm', 11), ('qae13.qasm', 13)]
    # circuits = [('qpe12.qasm', 12)]
    # circuits = [ ('vqe14.qasm', 14)]
    # circuits =  [ ('shor64.qasm', 64), ('qae33.qasm', 33), ('qae81.qasm', 81), ('mult64.qasm', 64), ('mult16.qasm', 16), ('heisenberg64.qasm', 64),  ('add17.qasm', 17), ('tfim400.qasm', 400)]
    # circuits =  [ ('vqe12.qasm', 12), ('vqe14.qasm', 14), ('qae33.qasm', 33), ('qae81.qasm', 81), ('mult64.qasm', 64), ('mult16.qasm', 16), ('heisenberg64.qasm', 64),  ('add17.qasm', 17)]
    # circuits =  [ ('qae33.qasm', 33), ('qae81.qasm', 81)]
    # circuits =  [ ('hub4.qasm', 4)]
    # circuits =  [ ('mult16.qasm', 16)]
    # circuits = [('add17.qasm', 17)]

    # circuits =  [ ('hhl8.qasm', 8)]
    # circuits =  [ ('shor26.qasm', 26)]
    # circuits =  [ ('qaoa5.qasm', 5)]
    # circuits =  [ ('qaoa5.qasm', 5)]
    # circuits = [('hhl8.qasm', 8),]
    # circuits =  [ ('adder9.qasm', 9)]
    # circuits = [('hhl8.qasm', 8)]
    circuits = [('heisenberg64.qasm', 64)]
    # circuits = [('mult64.qasm', 64)]


# ['QFACTOR-JAX', ] heisenberg7.qasm V
# ['QFACTOR-JAX', ] heisenberg8.qasm V
# ['QFACTOR-JAX', ] hub18.qasm V 
# ['QFACTOR-JAX', ] hub4.qasm V
# ['QFACTOR-JAX'] mult64.qasm V
# ['QFACTOR-JAX'] qae33.qasm V
# ['QFACTOR-JAX'] qae81.qasm V
# ['QFACTOR-JAX', ] qpe10.qasm V
# ['QFACTOR-JAX', ] qpe8.qasm V
# ['QFACTOR-JAX', ] tfim8.qasm V
# ['QFACTOR-JAX', ] vqe14.qasm V


# ['CERES_P', 'LBFGS_P', 'QFACTOR-RUST_P'] hhl8.qasm
# ['CERES_P', 'LBFGS_P', 'QFACTOR-RUST_P'] qpe12.qasm
    
    # instantiators = ['CERES', 'QFACTOR-RUST', 'LBFGS']
    # instantiators = ['CERES', 'QFACTOR-RUST', 'QFACTOR-JAX', 'LBFGS']
    # instantiators = ['CERES', 'QFACTOR-RUST', 'QFACTOR-JAX']
    # instantiators = ['CERES', 'QFACTOR-RUST',  'LBFGS']
    instantiators = ['CERES_P', 'QFACTOR-RUST_P',  'LBFGS_P']
    # instantiators = ['CERES', 'QFACTOR-RUST']
    # instantiators = ['CERES', 'QFACTOR-JAX']
    # instantiators = ['CERES']
    # instantiators = ['QFACTOR-JAX']
    # instantiators = ['QFACTOR-RUST']
    # instantiators = ['QFACTOR-RUST', 'LBFGS']
    # instantiators = ['CERES', 'LBFGS']
    # instantiators = ['CERES', 'CERES_P']
    # instantiators = ['CERES_P']
    # instantiators = ['LBFGS']

    
    time_limit = '11:00:00'
    partisions_size_l = [6, 7]
    # partisions_size_l = [3,4,5]
    # partisions_size_l = [3,4,5,6,7,8, 9,11,12]
    # partisions_size_l = [5, 6, 7,8, 9]
    # partisions_size_l = [13, 14, 15]

    num_multistarts_l = [32]

    seed_l = [None]


    # n_nodes = [4, 2, 1]
    n_nodes = [3]
    # n_nodes = [16]

    # n_workers_per_node = [4*3, 4*4, 4*5, 4*6, 4*7]
    # n_workers_per_node = [4*8, 4*10, 4*12, 4*14, 4*16]
    n_workers_per_node = [-1]
    n_amount_of_gpus_in_node=[0]

    # use_detached = False
    use_detached = True

    python_file = 'gate_deletion_perf_measurement.py'
    # python_file = 'gate_deletion_perf_measurement_test.py'

    for amount_of_gpus_per_node in n_amount_of_gpus_in_node:
        for workers_per_node in n_workers_per_node:
            for nodes in n_nodes:
                for cir_path, qubit_count in circuits:
                    for partitions_size in partisions_size_l:
                        if partitions_size > qubit_count:
                             continue
                        for seed in seed_l:               
                            for num_multistarts in num_multistarts_l:
                                for instantiator in instantiators:
                                    
                                    
                                
                                    create_and_run_a_job(cir_path,
                                                    partitions_size, python_file,
                                                    instantiator=instantiator,
                                                    num_multistarts=num_multistarts,
                                                    seed=seed, time_limit=time_limit,
                                                    nodes=nodes,
                                                    workers_per_node=workers_per_node,
                                                    amount_of_gpus_per_node=amount_of_gpus_per_node,
                                                    use_detached=use_detached)
