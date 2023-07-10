

import json
import subprocess
import time

from inst_test_utils import SlurmSubmissionsDb


job_templeae="""#!/bin/bash
#SBATCH --job-name={i}_{cir_name}_{n}_{instantiator}
#SBATCH -A m4141_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t {time}
#SBATCH -n 1
#SBATCH --mem=0
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --output={log_file_path}


date
uname -a
module load python
# conda activate my_env
conda activate dev_env

module load nvidia

echo "will run {env_vars} python  ./run_inst_test.py  {command_args} "

{env_vars} python  ./run_inst_test.py  {command_args}

"""


cpu_job_templeae="""#!/bin/bash
#SBATCH --job-name={i}_{cir_name}_{n}_{instantiator}
#SBATCH -A m4141
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t {time}
#SBATCH -n 1
#SBATCH --mem=0
#SBATCH --output={log_file_path}


date
uname -a
module load python
# conda activate my_env
conda activate dev_env

module load nvidia

echo "will run {env_vars} python  ./run_inst_test.py  {command_args} "

{env_vars} python  ./run_inst_test.py  {command_args}

"""




partitions_base_dir = '/global/homes/a/alonkukl/part_stats/partitions_qasms'
test_suite_path = '/global/homes/a/alonkukl/part_stats/partitions_qasms/test_partitions_10_rand.json'

multistarts = 32
diff_tol_a = 0.0   # Stopping criteria for distance change
diff_tol_r = 1e-5    # Relative criteria for distance change
dist_tol = 1e-10     # Stopping criteria for distance
max_iters = 1000000   # Maximum number of iterations
min_iters = 0     # Minimum number of iterations
diff_tol_step = 100
diff_tol_step_r = 0.1
beta = 0

insts =['QFACTOR-RUST', 'QFACTOR-JAX','LBFGS', 'CERES']
                            
job_script_name = 'inst_job.sh'
submission_db_name = 'inst_test10_rand.sqlite'
sleep_time = 0.05                          

with open(test_suite_path, 'r') as f:
    test_suite = json.load(f)

sdb = SlurmSubmissionsDb(db_name=submission_db_name)

results = []

i = 0
already_sent_skiped = 0
for qubits_count, circuits_dict in test_suite.items():
    if int(qubits_count) < 9:
        continue
    for circuit_name, partitions_to_test in circuits_dict.items():
        # if circuit_name != 'qpe12':
            # continue
        for block_name in partitions_to_test:
            qasm_file_path = f'{partitions_base_dir}/{circuit_name}.qasm.{qubits_count}/{block_name}'
            # for inst_name in ['CERES', 'LBFGS', 'QFACTOR-RUST', 'QFACTOR-JAX']:
            # for inst_name in ['CERES', 'LBFGS', 'QFACTOR-RUST', 'CERES_P', 'LBFGS_P', 'QFACTOR-RUST_P', 'QFACTOR-JAX']:
            for inst_name in ['CERES_P', 'LBFGS_P', 'QFACTOR-RUST_P', 'QFACTOR-JAX']:
            # for inst_name in ['CERES']:
                if 'JAX' in inst_name:
                    t = job_templeae
                else:
                    t = cpu_job_templeae
                time_limit = '02:10:00'
                to_write = open(job_script_name, 'w')
                i += 1
                job_name = f'{circuit_name}_{block_name}_{qubits_count}q_{inst_name}_{multistarts}_{dist_tol}'
                command_args = f"--input_qasm {qasm_file_path} --instantiator {inst_name} --partitions_size {qubits_count} --dist_tol {dist_tol}"
                other_inst_args = {'multistarts': multistarts}
                env_vars = ""
                if 'QFACTOR' in inst_name:
                    other_inst_args.update({ 'diff_tol_a' :diff_tol_a, 'diff_tol_r': diff_tol_r,'max_iters' :max_iters, 'min_iters': min_iters})
                    job_name +=f'_{diff_tol_a}_{diff_tol_r}_{max_iters}_{min_iters}'
                    if 'JAX' in inst_name:
                        other_inst_args.update({'beta':beta, 'diff_tol_step':diff_tol_step, 'diff_tol_step_r':diff_tol_step_r})
                        job_name += f'_{beta}_{diff_tol_step}_{diff_tol_step_r}'
                        env_vars += " XLA_PYTHON_CLIENT_PREALLOCATE=false"
                    else:
                        pass
                        # time_limit = '00:50:00'
                
                for k,v in other_inst_args.items():
                    command_args += f' --{k} {v}'

                log_path = f'/global/homes/a/alonkukl/Repos/bqskit/inst_tests_logs_10/{circuit_name}_{block_name}_{qubits_count}q_{inst_name}_inst_test_nodes_{multistarts}_starts-%j.txt'
                
                job_name += f'_{time_limit}'

                if sdb.exists(job_name):
                    already_sent_skiped+=1
                    continue

                to_write.write(t.format(i=i, cir_name=circuit_name, n=qubits_count, instantiator=inst_name,
                                                time=time_limit,  log_file_path=log_path, env_vars=env_vars, command_args=command_args))
                to_write.close()
                time.sleep(2*sleep_time)

                output = subprocess.check_output(['sbatch' , job_script_name])

                output_str = output.decode('utf-8')  # Decode the byte string output to a regular string
                lines = output_str.split('\n')  # Split the string into lines
                jobid = int(lines[0].split()[-1])
                
                print(f"Sent {job_name} and got {jobid}")
                log_path = log_path.replace('%j', str(jobid))
                sdb.add_submission(name= job_name, qubit_count=qubits_count, orig_circ_name=circuit_name, path_to_block=qasm_file_path, inst_name=inst_name,
                                    dist_tol=dist_tol, slurm_job_id=jobid, slurm_log_file_path=log_path, inst_params=other_inst_args)


                
print(f'Out of totoal possible {i} jobs, skipped {already_sent_skiped} and submitted {i-already_sent_skiped}')


    