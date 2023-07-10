#%%
import os
import random
import re


dir_path = '/global/homes/a/alonkukl/part_stats/partitions_qasms'
os.chdir(dir_path)
random.seed(314)

# set the range of values for i
i_range = range(3, 13)

# set the number of files to randomly choose from each directory
num_files = 10

# create an empty dictionary to store the chosen files
chosen_files = {}

# loop over the directories in the current directory
for circuit_dir in os.listdir('.'):
    # use regex to match the directory name to the pattern "<circuit>.qasm.<n>"
    match = re.match(r'^(\D+)(\d+)\.qasm\.(\d+)$', circuit_dir)
    if match:
        circuit_name = match.group(1)
        circuit_qubit_count = int(match.group(2))
        n = int(match.group(3))
        # check if the value of n is in the desired range
        if n in i_range and n != circuit_qubit_count:
            # create an empty dictionary to store the chosen files for the current i and circuit_name
            if n not in chosen_files:
                chosen_files[n] = {}
            # create an empty list to store the files in the directory that match the condition
            matching_files = []
            # loop over the files in the directory
            for filename in os.listdir(circuit_dir):
                # check if the file name matches the pattern "block_<number>.qasm"
                if filename.startswith('block_') and filename.endswith('.qasm'):
                    # read the third line of the file
                    with open(os.path.join(circuit_dir, filename)) as f:
                        first_line = f.readline()
                        second_line = f.readline()
                        third_line = f.readline().strip()
                    # check if the third line matches the desired structure
                    if third_line == f"qreg q[{n}];":
                        # add the file to the list of matching files
                        matching_files.append(filename)
            # choose 5 random files from the matching files list
            if len(matching_files) > num_files:
                chosen_files[n][f'{circuit_name}{circuit_qubit_count}'] = random.sample(matching_files, num_files)
            else:
                chosen_files[n][f'{circuit_name}{circuit_qubit_count}'] = matching_files

# print the chosen files
print(chosen_files)
#%%

import json
with open('test_partitions_10_rand.json', 'w') as f:
    json.dump(chosen_files, f)


#%%
def count_items(d):
    total = 0
    for v in d.values():
        if isinstance(v, dict):
            total += count_items(v)
        elif isinstance(v, list):
            total += len(v)
    return total

print(count_items(chosen_files))