#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


filename = './del_gates_4_3.csv'
filename = './del_gates_4_5.csv'
filename = './del_gates_4_7.csv'
filename = './del_gates_4_10.csv'
filename = './del_gates_4_12.csv'
filename = './big_run2.csv'
# filename = './qce23_small_circ.csv'
filename = './all_stats_no_qaoa12.csv'


orig_circuits_props = {
                'hub4.qasm'       :{'1Q_COUNT': 155,  '2Q_COUNT':180},
                'qaoa5.qasm'      :{'1Q_COUNT': 27,   '2Q_COUNT':42},
                'grover5_u3.qasm' :{'1Q_COUNT': 80,   '2Q_COUNT':48},
                'grover5.qasm' :{'1Q_COUNT': 80,   '2Q_COUNT':48},
                'adder9_u3.qasm'  :{'1Q_COUNT': 64,   '2Q_COUNT':98},
                'adder9.qasm'  :{'1Q_COUNT': 64,   '2Q_COUNT':98},
                'hub18.qasm'      :{'1Q_COUNT': 1992, '2Q_COUNT':3541},
                'adder63_u3.qasm' :{'1Q_COUNT': 2885, '2Q_COUNT':1405},
                'adder63.qasm' :{'1Q_COUNT': 2885, '2Q_COUNT':1405},
                'shor26.qasm'     :{'1Q_COUNT': 20896,'2Q_COUNT':21072},

                'add17.qasm'     :{'1Q_COUNT': 348,'2Q_COUNT':232},
                'heisenberg7.qasm':{'1Q_COUNT': 490,'2Q_COUNT':360},
                'heisenberg8.qasm':{'1Q_COUNT': 570,'2Q_COUNT':420},
                'heisenberg64.qasm':{'1Q_COUNT': 5050,'2Q_COUNT':3780},
                'hhl8.qasm'     :{'1Q_COUNT': 3288,'2Q_COUNT':2421},
                'mult8.qasm'     :{'1Q_COUNT': 210,'2Q_COUNT':188},
                'mult16.qasm'     :{'1Q_COUNT': 1264,'2Q_COUNT':1128},
                'mult64.qasm'     :{'1Q_COUNT': 61600,'2Q_COUNT':54816},
                'qae13.qasm'     :{'1Q_COUNT': 247,'2Q_COUNT':156},
                'qae11.qasm'     :{'1Q_COUNT': 176,'2Q_COUNT':110},
                'qae33.qasm'     :{'1Q_COUNT': 1617,'2Q_COUNT':1056},
                'qae81.qasm'     :{'1Q_COUNT': 7341,'2Q_COUNT':4840},
                'qpe8.qasm'     :{'1Q_COUNT': 519,'2Q_COUNT':372},
                'qpe10.qasm'     :{'1Q_COUNT': 1681,'2Q_COUNT':1260},
                'qpe12.qasm'     :{'1Q_COUNT': 3582,'2Q_COUNT':2550},
                'tfim16.qasm'     :{'1Q_COUNT': 916,'2Q_COUNT':600},
                'tfim8.qasm'     :{'1Q_COUNT': 428,'2Q_COUNT':280},
                'vqe14.qasm'     :{'1Q_COUNT': 10792,'2Q_COUNT':20392},
                'vqe12.qasm'     :{'1Q_COUNT': 4157,'2Q_COUNT':7640},
                'vqe5.qasm'     :{'1Q_COUNT': 132,'2Q_COUNT':91},
                'tfim400.qasm'     :{'1Q_COUNT': 88235,'2Q_COUNT':87670},






                    }



#%%



# Read in the CSV file using pandas, and specify the column names manually
column_names = ['True', 'INST', 'FILE_NAME', 'MULTISTARTS', 'PARTITION_SIZE', 'QUBIT_COUNT', 'RUNTIME', '1Q_COUNT', '2Q_COUNT', 'NODES', 'WORKERS_PER_NODE', 'GPUS']
data = pd.read_csv(filename, names=column_names)


for q in [1,2]:
    data[f'{q}Q_REDUCTION'] = None
    for index, row in data.iterrows():
        file_name = row['FILE_NAME']    
        base_line = orig_circuits_props[file_name][f'{q}Q_COUNT']

        reduction = 100 * (base_line - row[f'{q}Q_COUNT'])/base_line
        data.at[index, f'{q}Q_REDUCTION'] = reduction
    data[f'{q}Q_REDUCTION'] = data[f'{q}Q_REDUCTION'].astype('float64')

data[f'GATES_REDUCTION'] = None
for index, row in data.iterrows():
    file_name = row['FILE_NAME']    
    base_line = sum(orig_circuits_props[file_name][f'{q}Q_COUNT'] for q in [1,2])

    reduction = 100 * (base_line - sum(row[f'{q}Q_COUNT'] for q in [1,2]))/base_line
    data.at[index, f'GATES_REDUCTION'] = reduction
data[f'GATES_REDUCTION'] = data[f'GATES_REDUCTION'].astype('float64')

# Group the data by FILE_NAME, INST, and PARTITION_SIZE, and compute the mean of the RUNTIME column
grouped_data = data.groupby(['FILE_NAME', 'INST', 'PARTITION_SIZE'], as_index=False)['RUNTIME', '2Q_COUNT', '1Q_COUNT', '1Q_REDUCTION', '2Q_REDUCTION', 'GATES_REDUCTION'].mean()
#%%
# Loop through each unique FILE_NAME
for file_name in grouped_data['FILE_NAME'].unique():
    # if 'tfim' in file_name in ['heisenberg8.qasm']:
    if not 'qae' in file_name:
        continue
    # Subset the data for the current FILE_NAME
    file_data = grouped_data[grouped_data['FILE_NAME'] == file_name]
    circuit_name = file_name.split('.')[0]

    fig, ax = plt.subplots()

    # Loop through each unique INST for the current FILE_NAME
    for inst in file_data['INST'].unique():
        # Subset the data for the current INST
        inst_data = file_data[file_data['INST'] == inst]

        # Create a line plot for the current INST
        ax.plot(inst_data['PARTITION_SIZE'], inst_data['RUNTIME'], label=inst, marker='o', markersize=6)

    # Set the title and axis labels for the current plot
    ax.set_title(f'{circuit_name} Compile Time')
    ax.set_xlabel('Partition Size')
    ax.set_ylabel('Compile Time [s] (log scale)')

    # Set the y-axis to a logarithmic scale
    ax.set_yscale('log')
    ax.set_ylim(1)

    # Set the x-axis ticks to integers
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # Add a legend to the plot
    ax.legend()

    # Show the plot
    # plt.show()
    plt.savefig(circuit_name + '_run_time.pdf', bbox_inches='tight')

    num_insts = file_data['INST'].nunique()

    # Loop through each unique INST for the current FILE_NAME
    colors = plt.cm.tab10.colors[:num_insts]

    # Set the width of each bar based on the number of unique INSTs
    bar_width = 0.8 / num_insts

    for q in [1,2]:
        fig, ax = plt.subplots()
        bars_l = []
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        max_bar_value = 0

        for i, inst in enumerate(file_data['INST'].unique()):
            # Subset the data for the current INST
            inst_data = file_data[file_data['INST'] == inst]

            # Calculate the x-coordinates for the current INST's bars
            x_pos = np.arange(min(inst_data['PARTITION_SIZE']) + i * bar_width - (bar_width * (num_insts - 1) / 2) , max(inst_data['PARTITION_SIZE']) + 0.5 + i * bar_width )
            
            max_bar_value = max(max_bar_value, max(inst_data[f'{q}Q_REDUCTION']) )
            bars_l.append(ax.bar(x_pos, inst_data[f'{q}Q_REDUCTION'], width=bar_width, label=inst, color=plt.cm.tab10(i), hatch='/', edgecolor='black'))

    
        
        offset = max_bar_value / 100
        if q==1:
            ax.set_ylabel(f'U3 Gate Reduction [%]')
            
        else:
            # ax.set_ylim(0, 10)
            ax.set_ylabel(f'CNOT Gate Reduction [%]')
            # offset = max(0.5, max_bar_value / 100)

        for bars in bars_l:
            for bar in bars:
                h = bar.get_height()
                if inst_data['PARTITION_SIZE'].nunique() > 3:
                    r = int(h)
                elif h ==0:
                    r = 0
                elif h < 0.1:
                    r = f"{h:.0e}"
                else:
                    r = round(h,1)

                ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + offset,
                    r,
                    horizontalalignment='center',
                    fontsize=7,
                    # weight='bold',
                    )

        ax.set_xlabel('Partition Size')

        ax.legend()
        
  
        if q==1:
            ax.set_title(f'U3 Reduction in {circuit_name}')
            plt.savefig(circuit_name + '_u3_reduction.pdf', bbox_inches='tight')
        else:
            ax.set_title(f'CNOT Reduction in {circuit_name}')
            plt.savefig(circuit_name + '_cnot_reduction.pdf', bbox_inches='tight')


# %%
# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np

# # Read in the CSV file using pandas, and specify the column names manually
# column_names = ['True', 'INST', 'FILE_NAME', 'MULTISTARTS', 'PARTITION_SIZE', 'QUBIT_COUNT', 'RUNTIME', '1Q_COUNT', '2Q_COUNT', 'NODES', 'WORKERS_PER_NODE', 'GPUS']
# data = pd.read_csv('./del_gates_4_3.csv', names=column_names)

# # Group the data by FILE_NAME, INST, and PARTITION_SIZE, and compute the mean of the RUNTIME column
# grouped_data = data.groupby(['FILE_NAME', 'INST', 'PARTITION_SIZE'], as_index=False)['RUNTIME', '2Q_COUNT', '1Q_COUNT'].mean()

# # Loop through each unique FILE_NAME
# for file_name in grouped_data['FILE_NAME'].unique():
#     # Subset the data for the current FILE_NAME
#     file_data = grouped_data[grouped_data['FILE_NAME'] == file_name]

#     # Create a new figure and axis for the current FILE_NAME
#     fig, ax1 = plt.subplots()

    

#     num_insts = len(file_data['INST'].unique())

#     # Loop through each unique INST for the current FILE_NAME
#     colors = plt.cm.tab10.colors[:num_insts]

#     # Set the width of each bar based on the number of unique INSTs
#     bar_width = 0.8 / num_insts


#     for i, inst in enumerate(file_data['INST'].unique()):
#         # Subset the data for the current INST
#         inst_data = file_data[file_data['INST'] == inst]

#         # Calculate the x-coordinates for the current INST's bars
#         x_pos = np.arange(min(inst_data['PARTITION_SIZE']) + i * bar_width - (bar_width * (num_insts - 1) / 2) , max(inst_data['PARTITION_SIZE']) + 0.5 + i * bar_width )
        

#         # Create a bar plot for the 2Q_COUNT data
#         ax1.bar(x_pos, inst_data['2Q_COUNT'], width=bar_width, label=inst, color=plt.cm.tab10(i))

#         ax1.set_xlabel('PARTITION_SIZE')
#         ax1.set_ylabel('2Q_COUNT')

#         # Set the y-axis label color for the bar plot
#         ax1.tick_params(axis='y')

#     ax1.legend()
#     ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
#     # Show the plot
#     plt.show()

#     fig, ax2 = plt.subplots()

    
    

#     for i, inst in enumerate(file_data['INST'].unique()):
#         # Subset the data for the current INST
#         inst_data = file_data[file_data['INST'] == inst]
#         # Create a line plot for the current INST
#         ax2.plot(inst_data['PARTITION_SIZE'], inst_data['RUNTIME'], label=inst, marker='o', markersize=6)

#     # Set the title and axis labels for the current plot
#     ax2.set_title(file_name)
#     ax2.set_ylabel('RUNTIME (log scale)')

#     # Set the y-axis to a logarithmic scale
#     ax2.set_yscale('log')

#     # Add a legend to the plot
#     ax2.legend()
#     ax2.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
#     plt.show()    

# # %%

# %%
max_2q_reduction_index = [d.idxmax() for g,d in data.groupby(['FILE_NAME', 'INST', 'PARTITION_SIZE'], as_index=False)['2Q_REDUCTION']]
max_2q_reduction_rows = data.loc[max_2q_reduction_index]

inst_partition_mean_2q_reduction = max_2q_reduction_rows.groupby(['INST', 'PARTITION_SIZE'])['2Q_REDUCTION'].mean()
inst_partition_mean_1q_reduction = max_2q_reduction_rows.groupby(['INST', 'PARTITION_SIZE'])['1Q_REDUCTION'].mean()

all_jobs_filename = 'all_submitted_jobs.csv'
all_jobs_filename = 'all_submitted_jobs_may_2nd.csv'
jobs_df = pd.read_csv(all_jobs_filename, delimiter='_', names=['Circuit', 'Partition size', 'Instantiator'])

jobs_set = {'CERES_P':{}, 'LBFGS_P':{}, 'QFACTOR-RUST_P':{}, 'QFACTOR-JAX':{}}
for name, d in jobs_df.groupby(['Partition size', 'Instantiator']):
    size = int(name[0][:-1])
    inst = name[1]
    if not 'JAX'  in inst:
        inst += '_P'
    jobs_set[inst][size] = set(d['Circuit'].unique())
    
    
intersec = set()
for p_s, qf_s in jobs_set['QFACTOR-JAX'].items():
    if p_s in jobs_set['CERES_P']:
        c_s = jobs_set['CERES_P'][p_s]
        intersec = intersec.union(c_s.intersection(qf_s))



# %%


sns.lineplot(data=inst_partition_mean_2q_reduction.reset_index(), x= 'PARTITION_SIZE', y='2Q_REDUCTION', hue='INST')
plt.savefig('all_data_2q_reduction.pdf', bbox_inches='tight')
#%%
sns.lineplot(data=inst_partition_mean_1q_reduction.reset_index(), x= 'PARTITION_SIZE', y='1Q_REDUCTION', hue='INST')
plt.savefig('all_data_1q_reduction.pdf', bbox_inches='tight')
# %%


# max_2q_reduction_index = [d.idxmax() for g,d in data.groupby(['FILE_NAME', 'PARTITION_SIZE'], as_index=False)['2Q_REDUCTION']]

# max_2q_reduction_rows = data.loc[max_2q_reduction_index]

# inst_partition_mean_2q_reduction = max_2q_reduction_rows.groupby(['PARTITION_SIZE'])['2Q_REDUCTION'].mean()
# inst_partition_mean_1q_reduction = max_2q_reduction_rows.groupby(['PARTITION_SIZE'])['1Q_REDUCTION'].mean()

# sns.lineplot(data=inst_partition_mean_2q_reduction.reset_index(), x= 'PARTITION_SIZE', y='2Q_REDUCTION', label='CNOT')
# sns.lineplot(data=inst_partition_mean_1q_reduction.reset_index(), x= 'PARTITION_SIZE', y='1Q_REDUCTION', label='U3')

# %%

possible_insts = set(['QFACTOR-JAX', 'QFACTOR-RUST_P', 'CERES_P', 'LBFGS_P'])
possible_insts = set(['QFACTOR-JAX',  'CERES_P'])

gates_reduction_sum = {}

for low,high in [(3,9), (10, 35), (36, 1000)]:
    gates_reduction_sum[(low,high)] = {i:{} for i in possible_insts}

    for g,d in max_2q_reduction_rows[['INST' ,'FILE_NAME', 'PARTITION_SIZE', '1Q_REDUCTION' ,'2Q_REDUCTION', 'QUBIT_COUNT', 'GATES_REDUCTION']].groupby(['FILE_NAME' ,'PARTITION_SIZE']):
        file, par = g
        if not file.split('.')[0] in intersec:
            continue
        if d['QUBIT_COUNT'].iloc[0]<low or  d['QUBIT_COUNT'].iloc[0]>high:
            continue
        for inst, inst_data in d.groupby('INST'):
            if not inst in possible_insts:
                continue
            dinst = gates_reduction_sum[(low,high)][inst].get(par, {'U3':0, 'CNOT':0, 'amount':0, 'Gates':0, 'Finished':0})
            dinst['amount'] += 1
            dinst['U3'] +=float(inst_data['1Q_REDUCTION'])
            dinst['CNOT'] +=float(inst_data['2Q_REDUCTION'])
            dinst['Gates'] += float(inst_data['GATES_REDUCTION'])
            dinst['Finished'] += 1
            # if 'JAX' in inst and par==3:
            #     print(file)
        
            gates_reduction_sum[(low,high)][inst][par]= dinst
        for inst in possible_insts-set(d.INST.unique()):
            # if 'JAX' in inst and par==3:
            #     print(file)
            # if data[(data['INST']==inst) & (data['FILE_NAME'] == file)].size == 0:
                # print(inst, file, par)
                # continue
            dinst = gates_reduction_sum[(low,high)][inst].get(par, {'U3':0, 'CNOT':0, 'amount':0, 'Gates':0, 'Finished':0})
            dinst['amount'] += 1
            gates_reduction_sum[(low,high)][inst][par]= dinst

gates_reduction_avg = []

for k in gates_reduction_sum.keys():
    
    for inst in gates_reduction_sum[k].keys():
        for par, d in gates_reduction_sum[k][inst].items():
            u3_sum = d['U3']
            cnot_sum = d['CNOT']
            gates_sum = d['Gates']
            amount_sum = d['amount']
            fin_amount = d['Finished']
            gates_reduction_avg.append({'Compliation rate':f'{fin_amount}/{amount_sum}','Qubit range':k ,'Instantiator':inst, 'Parition size':par, 'U3 reduction [%]':u3_sum/amount_sum, 'CNOT reduction [%]':cnot_sum/amount_sum, 'Total gates reduction [%]':gates_sum/amount_sum})


gates_reduction_df = pd.DataFrame(gates_reduction_avg)

# %%
import seaborn as sns
import matplotlib.pyplot as plt

# create the line plot
plot = sns.lineplot(data=gates_reduction_df, x='Parition size', y='U3 reduction [%]', style='Instantiator', hue='Qubit range', legend='full')

# customize the plot

handles, labels = plot.get_legend_handles_labels()
labels = labels[-7:]
handles = handles[-7:]
labels[6] = 'QF-JAX'
plot.legend(handles, labels, loc='center right' )
plot.set_xticks(gates_reduction_df['Parition size'].astype(int))
# plot2= sns.scatterplot(data=gates_reduction_df, x='PARTITION_SIZE', y='U3_reduction',  style='INST', hue='Qubit range', markers=True, ax=plot, legend=False)

# show the plot
# plt.show()

# sns.lineplot(data=gates_reduction_df, x='PARTITION_SIZE', y='U3_reduction', hue='INST')
plt.savefig('all_data_1q_reduction_per_circuit_qubits.pdf', bbox_inches='tight')
# %%
plot = sns.lineplot(data=gates_reduction_df, x='Parition size', y='CNOT reduction [%]',   style='Instantiator', hue='Qubit range')
# customize the plot

# sns.scatterplot(data=gates_reduction_df, x='PARTITION_SIZE', y='CNOT_reduction',  hue=['INST', 'Qubit range'], markers=True, ax=plot, legend=False)
handles, labels = plot.get_legend_handles_labels()
labels = labels[-7:]
handles = handles[-7:]
labels[6] = 'QF-JAX'
plot.legend(handles, labels, loc='upper left' )
plot.set_xticks(gates_reduction_df['Parition size'].astype(int))

# show the plot
# plt.show()

plt.savefig('all_data_2q_reduction_per_circuit_qubits.pdf', bbox_inches='tight')


#%%
for inst, d in gates_reduction_df.groupby('Instantiator'):
    plt.figure()
    plot = sns.lineplot(data=d, x='Parition size', y='Total gates reduction [%]',   hue='Qubit range')
    plot.set(ylabel=f'Total gates reduction [%]')
    plot.set(title=inst)
    plot.set(ylim=(0,55))
    for _, row in d.iterrows():
        x, y = row['Parition size'], row['Total gates reduction [%]']
        if row['Qubit range'][0] == 3:
            # if x!= 3 and x!= 9:
            #     continue
            if inst == 'QFACTOR-JAX':
                yoffset = -10 if(x!=3) else 5
            else:
                yoffset = -10 if(x==4) else 5
        elif row['Qubit range'][0] == 10:
            if 'CERES_P' == inst:
                yoffset = -10 if x==7 else 7
            else:
                yoffset = -10
            # if x!= 3 and x!= 7:
            #     continue

        elif row['Qubit range'][0] == 36:
            yoffset = 7
            # if x!= 3 and x!= 7:
            #     continue

        label = f"{row['Compliation rate']}"

        plt.annotate(label, xy=(x, y), xytext=(-8, yoffset), textcoords='offset points')
    
    # for i, row in d.iterrows():
    #     x = row['Parition size']
    #     y = row['Total gates reduction [%]']
    #     cr = row['Compliation rate']
    #     plt.text(x, y, f"{cr}", horizontalalignment='center', verticalalignment='bottom')
    
    plt.savefig(f'{inst}_total_gates_reduction_in_buckets.pdf', bbox_inches='tight')

# %%



possible_insts = set(['QFACTOR-JAX', 'QFACTOR-RUST_P', 'CERES_P', 'LBFGS_P'])
# possible_insts = set(['QFACTOR-JAX',  'CERES_P'])

gates_reduction_sum = {i:{} for i in possible_insts}

for g,d in max_2q_reduction_rows[['INST' ,'FILE_NAME', 'PARTITION_SIZE', '1Q_REDUCTION' ,'2Q_REDUCTION', 'QUBIT_COUNT']].groupby(['FILE_NAME' ,'PARTITION_SIZE']):
    file, par = g
    if not file.split('.')[0] in intersec:
            continue
    # if d['QUBIT_COUNT'].iloc[0]<low or  d['QUBIT_COUNT'].iloc[0]>high:
    #     continue
    for inst, inst_data in d.groupby('INST'):
        
        if not inst in possible_insts:
            continue
        dinst = gates_reduction_sum[inst].get(par, {'U3':0, 'CNOT':0, 'amount':0})
        dinst['amount'] += 1
        dinst['U3'] +=float(inst_data['1Q_REDUCTION'])
        dinst['CNOT'] +=float(inst_data['2Q_REDUCTION'])
        gates_reduction_sum[inst][par]= dinst
    for inst in possible_insts-set(d.INST.unique()):
        if data[(data['INST']==inst) & (data['FILE_NAME'] == file)].size == 0:
            continue
        dinst = gates_reduction_sum[inst].get(par, {'U3':0, 'CNOT':0, 'amount':0})
        dinst['amount'] += 1
        gates_reduction_sum[inst][par]= dinst

gates_reduction_avg = []

    
for inst in gates_reduction_sum.keys():
    for par, d in gates_reduction_sum[inst].items():
        u3_sum = d['U3']
        cnot_sum = d['CNOT']
        amount_sum = d['amount']
        gates_reduction_avg.append({'Instantiator':inst, 'Parition size':par, 'U3 reduction [%]':u3_sum/amount_sum, 'CNOT reduction [%]':cnot_sum/amount_sum})


gates_reduction_df = pd.DataFrame(gates_reduction_avg)

# %%
import seaborn as sns
import matplotlib.pyplot as plt

# create the line plot
plot = sns.lineplot(data=gates_reduction_df, x='Parition size', y='U3 reduction [%]', hue='Instantiator', legend='full')

# customize the plot

handles, labels = plot.get_legend_handles_labels()
labels = [l.replace("QFACTOR", 'QF') for l in labels]
plot.legend(handles=handles ,labels=labels)
# plot.set(ylabel='U3 reduction [%]')
plot.set_xticks(gates_reduction_df['Parition size'].astype(int))
# plot.set(xlabel='Partition size')
# plot2= sns.scatterplot(data=gates_reduction_df, x='PARTITION_SIZE', y='U3_reduction',  style='INST', hue='Qubit range', markers=True, ax=plot, legend=False)

# show the plot
# plt.show()

# sns.lineplot(data=gates_reduction_df, x='PARTITION_SIZE', y='U3_reduction', hue='INST')
plt.savefig('all_data_1q_reduction.pdf', bbox_inches='tight')
# %%
plot = sns.lineplot(data=gates_reduction_df, x='Parition size', y='CNOT reduction [%]',   hue='Instantiator')
# customize the plot

handles, labels = plot.get_legend_handles_labels()
labels = [l.replace("QFACTOR", 'QF') for l in labels]
plot.legend(handles=handles ,labels=labels)

plot.set_xticks(gates_reduction_df['Parition size'].astype(int))


# show the plot
# plt.show()

plt.savefig('all_data_2q_reduction.pdf', bbox_inches='tight')

# %%

# %%
