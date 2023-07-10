#%%

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from inst_test_utils import  SlurmSubmissionsDb


# submission_db_name = 'inst_test1.sqlite'
submission_db_name = 'inst_test10_rand.sqlite'
sdb = SlurmSubmissionsDb(db_name=submission_db_name)
all_subs = list(sdb.get_all_submissions())
#%%

df = pd.DataFrame(all_subs)
df.drop_duplicates(subset=['inst_name', 'path_to_block', 'qubit_count'])
df['qubit_count'] = df['qubit_count'].astype(int)
# Compute success rate
df_success = df[(df["inst_succsed"] == True)]
success_rate = df_success.groupby(["inst_name", "qubit_count"]).size() / df.groupby(["inst_name", "qubit_count"]).size()
success_rate_df = success_rate.reset_index(name="success_rate")
# %%

# calculate total succus rate
print(df.groupby('inst_name')['inst_succsed'].mean())
# %%
print(df.groupby(['inst_name', 'inst_succsed'])['inst_time'].mean().unstack())
# %%

markers = {'CERES': 'D','CERES_P': 'D',
           'LBFGS': '+','LBFGS_P': '+',
           'QFACTOR-RUST': 'x','QFACTOR-RUST_P': 'x',
           'QFACTOR-JAX':'o'}

# group the data by circ_name, inst_name, and qubit_count, and calculate the mean of inst_time
avg_time_s = df[df["inst_succsed"] == True].groupby(['orig_circ_name', 'inst_name', 'qubit_count'])['inst_time'].mean()
avg_time = df.groupby(['orig_circ_name', 'inst_name', 'qubit_count'])['inst_time'].mean()

# create a new data frame with the result of the groupby operation
df_avg_time = avg_time.reset_index()
df_avg_time_s = avg_time_s.reset_index()

# loop over the unique circ_names
for circ_name in df_avg_time['orig_circ_name'].unique():
    # select the subset of data for the current circ_name

    # subset = df_avg_time[df_avg_time['orig_circ_name'] == circ_name]

    print(circ_name)
    for df_temp in [df_avg_time, df_avg_time_s]:
    # for df_temp in [df_avg_time]:
      subset = df_temp[df_temp['orig_circ_name'] == circ_name]
      if len(subset) == 0:
          continue
      # create a new plot for the current circ_name
      plt.figure()
      
      # loop over the unique combinations of inst_name for the current circ_name
      for inst_name in subset['inst_name'].unique():
          # select the subset of data for the current combination of circ_name and inst_name
          inst_subset = subset[subset['inst_name'] == inst_name]
          
          # plot the subset of data as a line plot with markers
          plt.plot(inst_subset['qubit_count'], inst_subset['inst_time'], label=f"{inst_name}", marker=markers[inst_name])
      
      # add labels and legend to the plot
      plt.xlabel('qubit_count')
      plt.ylabel('Average inst_time')
      plt.title(circ_name)
      plt.legend()

      # set x-axis to integers
      plt.xticks(range(min(subset['qubit_count']), max(subset['qubit_count'])+1))
      
      
      # # set y-axis to log scale
      # plt.yscale('log')


      plt.savefig(circ_name + '_runtime.pdf', bbox_inches='tight')
      
# show all the plots
plt.show()

# %%

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt


# # Filter out the rows where inst_succsed is NaN
# # Group the data by orig_circ_name, inst_name, and qubit_count
# grouped = df[df["inst_succsed"].notna()].groupby(["orig_circ_name", "inst_name", "qubit_count"])



# # Iterate over each orig_circ_name and create a separate plot for each one
# for orig_circ_name, group in grouped:
#     # Set up the plot
#     fig, ax = plt.subplots(figsize=(10, 8))
#     ax.set_xlabel("Qubit Count", fontsize=14)
#     ax.set_ylabel("Success Rate", fontsize=14)
#     ax.set_title(f"Success Rate vs Qubit Count for {orig_circ_name}", fontsize=16)

#     # Iterate over each inst_name and plot the data
#     for inst_name, inst_group in group.groupby("inst_name"):
#         # Calculate the mean success rate for each qubit count
#         # success_rate = inst_group["inst_succsed"].value_counts(normalize=True).loc[:, True]
#         success_rate = inst_group["inst_succsed"].value_counts(normalize=True).loc[True]

#         success_rate_mean = success_rate.groupby("qubit_count").mean()

#         # Convert the index to integers
#         success_rate_mean.index = success_rate_mean.index.astype(int)

#         # Plot the data as a scatter plot with a line connecting the points
#         ax.plot(success_rate_mean.index, success_rate_mean, "-o", label=inst_name)

#     # Add a legend and show the plot
#     ax.legend()
#     plt.show()


#%%
import matplotlib.pyplot as plt
import pandas as pd

hatches = {'CERES': '\\','CERES_P': '-',
           'LBFGS': '\\','LBFGS_P': '-',
           'QFACTOR-RUST': '\\','QFACTOR-RUST_P': '-',
           'QFACTOR-JAX':'x'}


# Compute success rate
df_success = df[df["inst_succsed"] == True]
success_rate = df_success.groupby(["orig_circ_name", "inst_name", "qubit_count"]).size() / df.groupby(["orig_circ_name", "inst_name", "qubit_count"]).size()
success_rate = success_rate.reset_index(name="success_rate")


num_insts = df['inst_name'].nunique()

# Loop through each unique INST for the current FILE_NAME
colors = plt.cm.tab10.colors[:num_insts]

# Set the width of each bar based on the number of unique INSTs
bar_width = 0.8 / num_insts

# Create a separate plot for each orig_circ_name
for orig_circ_name, df_orig in success_rate.groupby("orig_circ_name"):
    # Create a separate subplot for each inst_name
    fig, ax = plt.subplots()


    # for inst_name, df_inst in df_orig.groupby("inst_name"):
    for i, inst in enumerate(df_orig['inst_name'].unique()):
        df_inst = df_orig[df_orig['inst_name'] == inst]
        # Plot the success rate for each inst_name
        # Calculate the x-coordinates for the current INST's bars
        x_pos = np.arange(min(df_inst['qubit_count']) + i * bar_width - (bar_width * (num_insts - 1) / 2) , max(df_inst['qubit_count']) + 0.5 + i * bar_width )

        ax.bar(x_pos,  df_inst["success_rate"], width=bar_width, label=inst, color=plt.cm.tab10(i), hatch=hatches[inst], edgecolor='black')


    # Set the axis labels and title
    ax.set_xlabel("qubit_count")
    ax.set_ylabel("Success rate")
    ax.set_title(f"Success rate by inst_name for {orig_circ_name}")
    ax.set_ylim([0, 1.4])
    ax.set_xticks(df_orig["qubit_count"].unique())
    ax.legend()

    plt.savefig(orig_circ_name + '_success_rate.pdf', bbox_inches='tight')
plt.show()

# %%


temp = r"""
\begin{figure}[ht]
  \centering
  \begin{subfigure}{0.48\columnwidth}
    \includegraphics[width=\textwidth]{figures/inst_tests_results/%s_runtime.pdf}
    \label{subfig:%s_inst_rt}
  \end{subfigure}
  \hfill
  \begin{subfigure}{0.48\columnwidth}
    \includegraphics[width=\textwidth]{figures/inst_tests_results/%s_success_rate.pdf}
    \label{subfig:%s_inst_sr}
  \end{subfigure}
\end{figure}
"""

for circ_name in df['orig_circ_name'].unique():
    print(temp%(circ_name, circ_name,circ_name, circ_name))
# %%

# Filter out rows with NaN values in inst_time column
df_filtered = df.dropna(subset=['inst_time'])

# Group by qubit_count and inst_name, and sum inst_time
grouped = df_filtered.groupby(['qubit_count', 'inst_name'])['inst_time'].sum()
# grouped.loc[3] = grouped.loc[3].apply(lambda x: x / 4 if not np.isnan(x) else np.nan)


l = {}
# Normalize the inst_time values for each qubit_count with respect to inst_name 'QFACTOR-JAX'
for qubit_count in grouped.index.levels[0]:
    qfact_jax_sum = grouped.loc[(qubit_count, 'QFACTOR-JAX')]
    l[qubit_count] = grouped.loc[qubit_count] / qfact_jax_sum
    
  
# Print the resulting values
print(l)

# %%
import matplotlib.pyplot as plt

# Convert the dictionary into a list of tuples
data = list(l.items())

# Create a new figure
fig, ax = plt.subplots()

# Iterate over the inst_names
for i, inst_name in enumerate(df['inst_name'].unique()):
    # Get the x and y values for this inst_name
    x = []
    y = []
    for qubit_count, pds in l.items():
      if not inst_name in pds:
          continue
      x.append(qubit_count)
      y.append(pds[inst_name])
    # Add the data as a new series to the plot
    ax.plot(x, y, label=inst_name)

# Add labels and legend
ax.set_xlabel('qubit_count')
ax.set_ylabel('normalized inst_time')
ax.legend()

# Show the plot
plt.show()


#%%

import pandas as pd
import seaborn as sns




for filter_out_unsucsuful in [True, False]:

  def fff(row):
    p = row['path_to_block']
    jax_row = df.loc[(df['inst_name'] == 'QFACTOR-JAX') & (df['path_to_block'] == p)]
    if filter_out_unsucsuful:
      if jax_row['inst_succsed'].iloc[0] and row['inst_succsed']:
        return row['inst_time'] / jax_row['inst_time'].iloc[0]
      else:
        return None
    return row['inst_time'] / jax_row['inst_time'].iloc[0]    

    # create a new column inst_time_norm with normalized inst_time values
  df['inst_time_norm'] = df.apply(fff, axis=1)

  df_filtered = df.dropna(subset=['inst_time_norm'])


  sns.set_style('whitegrid')
  sns.set_palette('bright')
  plot = sns.lineplot(data=df_filtered, x='qubit_count', y='inst_time_norm', hue='inst_name', )
  # if filter_out_unsucsuful:
  #   plt.title('Normelized average instantiation time per amount of qubits in circuit\n filterd out unsucsuful runs')
  # else:
  #   plt.title('Normelized average instantiation time per amount of qubits in circuit')
  plt.xlabel('Qubit number')
  plt.ylabel('Normelized average instantiation time (log-scale)')
  plt.yscale('log')
  handles, labels = plot.get_legend_handles_labels()
  labels= [l.replace('QFACTOR', 'QF') for l in labels]
  # labels[0] = 'Instantiator'
  plot.legend(handles, labels, title='Instantiator' )
  plt.savefig(f'norm_avg_inst_time_testsuit_{filter_out_unsucsuful}.pdf', bbox_inches='tight')
  plt.show()

# %%

import matplotlib.pyplot as plt
import pandas as pd





num_insts = df['inst_name'].nunique()

# Loop through each unique INST for the current FILE_NAME
colors = plt.cm.tab10.colors[:num_insts]

# Set the width of each bar based on the number of unique INSTs
bar_width = 0.8 / num_insts



# Create a separate subplot for each inst_name
fig, ax = plt.subplots()


# for inst_name, df_inst in df_orig.groupby("inst_name"):
for i, inst in enumerate(success_rate_df['inst_name'].unique()):
    df_inst = success_rate_df[success_rate_df['inst_name'] == inst]
    rel = (df_inst.set_index('qubit_count')["success_rate"]/success_rate_df[success_rate_df['inst_name'] == 'QFACTOR-JAX'].set_index('qubit_count')["success_rate"]).fillna(0)
    
    ax.plot(df_inst['qubit_count'],  rel, label=inst)
    # ax.bar(df_inst['qubit_count'], df_inst.set_index('qubit_count')["success_rate"])


# Set the axis labels and title
ax.set_xlabel("qubit_count")
ax.set_ylabel("Success rate")
# ax.set_title(f"Success rate by inst_name for {orig_circ_name}")
ax.set_ylim([0, 1.4])
ax.set_xticks(success_rate_df["qubit_count"].unique())
ax.legend()

plt.savefig('total_success_rate.pdf', bbox_inches='tight')
plt.show()

# %%


# import pandas as pd
# import seaborn as sns




# # df_new = df.drop_duplicates(subset=['inst_name', 'path_to_block', 'qubit_count']).fillna({'inst_time':2*60*60} )
# df_new = df.drop_duplicates(subset=['inst_name', 'path_to_block', 'qubit_count']).copy()
# df_new = df_new[(df_new['inst_name'].isin(['CERES_P', 'QFACTOR-JAX', 'QFACTOR-RUST_P']))]




# def fff(row):
#   p = row['path_to_block']
#   # print(p)
#   jax_row = df_new.loc[(df_new['inst_name'] == 'QFACTOR-JAX') & (df_new['path_to_block'] == p)]    
#   return row['inst_time'] / jax_row['inst_time'].iloc[0]    


# df_new['inst_time_norm'] = df_new.apply(fff, axis=1)

# df_filtered = df_new.dropna(subset=['inst_time_norm'])


# fig, ax1 = plt.subplots()

# sns.set_style('whitegrid')
# sns.set_palette('bright')
# # plot = sns.lineplot(data=df_filtered, x='qubit_count', y='inst_time_norm', hue='inst_name', ax=ax1, legend=False )
# plot = sns.lineplot(data=df_filtered, x='qubit_count', y='inst_time_norm', hue='inst_name', ax=ax1 )
# # handles, labels = plot.get_legend_handles_labels()
# # labels= [l.replace('QFACTOR', 'QF') for l in labels]
# # plot.legend(handles, labels )
# # if filter_out_unsucsuful:
# #   plt.title('Normelized average instantiation time per amount of qubits in circuit\n filterd out unsucsuful runs')
# # else:
# #   plt.title('Normelized average instantiation time per amount of qubits in circuit')
# plt.xlabel('Qubit number')
# ax1.set_ylabel('Normelized average instantiation time (log-scale)')
# ax1.set_yscale('log')

# # plt.savefig(f'norm_avg_inst_time_testsuit_{filter_out_unsucsuful}.pdf', bbox_inches='tight')

# ax2 = ax1.twinx()
# # ax2 = ax1

# num_insts = 2

# # Loop through each unique INST for the current FILE_NAME
# colors = ['b', 'g']
# # colors = plt.cm.tab10.colors[0]  + plt.cm.tab10.colors[2]

# # Set the width of each bar based on the number of unique INSTs
# bar_width = 0.8 / num_insts

# for i, inst in  enumerate(['CERES_P',  'QFACTOR-RUST_P', 'QFACTOR-JAX']):
#     df_inst = success_rate_df[success_rate_df['inst_name'] == inst]
#     rel = (df_inst.set_index('qubit_count')["success_rate"]/success_rate_df[success_rate_df['inst_name'] == 'QFACTOR-JAX'].set_index('qubit_count')["success_rate"]).fillna(0)
    
#     ax2.plot(df_inst['qubit_count'],  rel, label=inst, color=colors[i])
#     x_pos = np.arange(min(df_inst['qubit_count']) + i * bar_width - (bar_width * (num_insts - 1) / 2) , max(df_inst['qubit_count']) + 0.5 + i * bar_width )

#     # ax2.bar(x_pos,  df_inst.set_index('qubit_count')["success_rate"], width=bar_width, label=inst, color=plt.cm.tab10(i), hatch=hatches[inst], edgecolor='black')
#     # ax2.bar(df_inst['qubit_count'], df_inst.set_index('qubit_count')["success_rate"])


# # Set the axis labels and title
# ax2.set_xlabel("qubit_count")
# ax2.set_ylabel("Success rate")
# # ax.set_title(f"Success rate by inst_name for {orig_circ_name}")
# ax2.set_ylim([0, 2])
# # ax2.set_yscale('log')
# ax2.set_xticks(success_rate_df["qubit_count"].unique())
# ax2.legend()

# plt.show()

# %%


import pandas as pd
import seaborn as sns




# df_new = df.drop_duplicates(subset=['inst_name', 'path_to_block', 'qubit_count']).fillna({'inst_time':10*60} )
df_new = df.drop_duplicates(subset=['inst_name', 'path_to_block', 'qubit_count']).copy()
df_new = df_new[(df_new['inst_name'].isin(['CERES_P', 'QFACTOR-JAX', 'QFACTOR-RUST_P']))]




def fff(row):
  p = row['path_to_block']
  # print(p)
  jax_row = df_new.loc[(df_new['inst_name'] == 'QFACTOR-JAX') & (df_new['path_to_block'] == p)]    
  return row['inst_time'] / jax_row['inst_time'].iloc[0]    


df_new['inst_time_norm'] = df_new.apply(fff, axis=1)

df_filtered = df_new.dropna(subset=['inst_time_norm'])


fig, ax1 = plt.subplots()

colors = ['b','m', 'g']
sns.set_style('whitegrid')
sns.set_palette('bright')
plot = sns.lineplot(data=df_filtered, x='qubit_count', y='inst_time_norm', hue='inst_name', ax=ax1, legend=True, palette=colors )
# if filter_out_unsucsuful:
#   plt.title('Normelized average instantiation time per amount of qubits in circuit\n filterd out unsucsuful runs')
# else:
#   plt.title('Normelized average instantiation time per amount of qubits in circuit')
handles, labels = plot.get_legend_handles_labels()
labels= [l.replace('QFACTOR', 'QF') for l in labels]
plot.legend(handles, labels, title='')
plt.xlabel('Qubit number')
ax1.set_ylabel('Normelized average instantiation time (log-scale)')
ax1.set_yscale('log')
ax1.set_ylim(top=1e2)
# plt.savefig(f'norm_avg_inst_time_testsuit_{filter_out_unsucsuful}.pdf', bbox_inches='tight')

ax2 = ax1.twinx()

num_insts = 2

# Loop through each unique INST for the current FILE_NAME

# colors = plt.cm.tab10.colors[0]  + plt.cm.tab10.colors[2]

# Set the width of each bar based on the number of unique INSTs
bar_width = 0.8 / num_insts

for i, inst in  enumerate(['CERES_P', 'QFACTOR-JAX',   'QFACTOR-RUST_P']):
    df_inst = success_rate_df[success_rate_df['inst_name'] == inst]
    rel = (df_inst.set_index('qubit_count')["success_rate"]/success_rate_df[success_rate_df['inst_name'] == 'QFACTOR-JAX'].set_index('qubit_count')["success_rate"]).fillna(0)
    
    ax2.plot(df_inst['qubit_count'],  rel, '--', label=inst, color=colors[i])
    x_pos = np.arange(min(df_inst['qubit_count']) + i * bar_width - (bar_width * (num_insts - 1) / 2) , max(df_inst['qubit_count']) + 0.5 + i * bar_width )

    # ax2.bar(x_pos,  df_inst.set_index('qubit_count')["success_rate"], width=bar_width, label=inst, color=plt.cm.tab10(i), hatch=hatches[inst], edgecolor='black')
    # ax2.bar(df_inst['qubit_count'], df_inst.set_index('qubit_count')["success_rate"])


# Set the axis labels and title
ax2.set_xlabel("qubit_count")
ax2.set_ylabel("Normelized average success rate")
# ax.set_title(f"Success rate by inst_name for {orig_circ_name}")
ax2.set_ylim([0, 2])
# ax2.set_yscale('log')
ax2.set_xticks(success_rate_df["qubit_count"].unique())
handles, labels = ax2.get_legend_handles_labels()
labels= [l.replace('QFACTOR', 'QF') for l in labels]
ax2.legend(labels=labels, title='')
# ax2.legend()
plt.savefig('rel_inst_time_with_sr.pdf')
plt.show()

# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



sorted_qubit_count = sorted(df_success['qubit_count'].unique())
sns.violinplot(data=df_success, x='qubit_count', y='inst_time', order=sorted_qubit_count)
plt.yscale('log')

plt.show()

# %%
sorted_qubit_count = sorted(df_success['qubit_count'].unique())

# Iterate over unique 'inst_type' values
for inst_name in df_success['inst_name'].unique():
    # Filter the dataframe for each 'inst_type'
    filtered_df = df_success[df_success['inst_name'] == inst_name]
    
    # Create the violin plot for the filtered dataframe
    sns.violinplot(data=filtered_df, x='qubit_count', y='inst_time', order=sorted_qubit_count)
    
    # Set the y-axis scale to logarithmic
    # plt.yscale('log')
    
    # Set the title for each plot based on 'inst_name'
    plt.title(f'Violin Plot: {inst_name}')
    
    # Display the plot
    plt.show()

# %%
