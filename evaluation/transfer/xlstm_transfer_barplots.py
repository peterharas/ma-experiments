import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
df_base = pd.read_csv('results/BASELINE_results_20260427_080304.csv')
df_base_lk = pd.read_csv('results/BASELINE_LASTKNOWN_results_20260427_080725.csv')
df_xlstm = pd.read_csv('results/xLSTM_results_20260608_090620.csv')
df_xlstm_large = pd.read_csv('results/xLSTM_LARGE_results_20260612_065528.csv')
df_xlstm_transfer = pd.read_csv('results/xLSTM_TRANSFER_results_20260716_090233.csv')

# Get the 7 springs from xLSTM_TRANSFER
transfer_springs = sorted(df_xlstm_transfer['spring_id'].unique())

# Filter all dataframes to ensure we only plot these 7 springs
df_base = df_base[df_base['spring_id'].isin(transfer_springs)]
df_base_lk = df_base_lk[df_base_lk['spring_id'].isin(transfer_springs)]
df_xlstm = df_xlstm[df_xlstm['spring_id'].isin(transfer_springs)]
df_xlstm_large = df_xlstm_large[df_xlstm_large['spring_id'].isin(transfer_springs)]

# Setup a 2x2 grid for the 4 horizons
fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharey=True)
axes = axes.flatten()

horizons = [1, 2, 3, 4]
# You can change these colors to anything you prefer later
colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] 

x = np.arange(len(transfer_springs))
width = 0.25

for i, h in enumerate(horizons):
    ax = axes[i]
    
    # Extract data for the current horizon and align by spring_id
    h_xlstm = df_xlstm[df_xlstm['horizon'] == h].set_index('spring_id')['nse'].reindex(transfer_springs).values
    h_xlstm_l = df_xlstm_large[df_xlstm_large['horizon'] == h].set_index('spring_id')['nse'].reindex(transfer_springs).values
    h_xlstm_t = df_xlstm_transfer[df_xlstm_transfer['horizon'] == h].set_index('spring_id')['nse'].reindex(transfer_springs).values
    
    h_base = df_base[df_base['horizon'] == h].set_index('spring_id')['nse'].reindex(transfer_springs).values
    h_base_lk = df_base_lk[df_base_lk['horizon'] == h].set_index('spring_id')['nse'].reindex(transfer_springs).values
    
    # Plot grouped bars
    ax.bar(x - width, h_xlstm, width, label='xLSTM', color=colors[0], alpha=0.8)
    ax.bar(x, h_xlstm_l, width, label='xLSTM_LARGE', color=colors[1], alpha=0.8)
    ax.bar(x + width, h_xlstm_t, width, label='xLSTM_TRANSFER', color=colors[2], alpha=0.8)
    
    # Plot lines for baselines
    ax.plot(x, h_base, color='dimgray', linestyle='-', marker='o', linewidth=2, label='BASELINE')
    ax.plot(x, h_base_lk, color='silver', linestyle='--', marker='x', linewidth=2, label='BASELINE_LASTKNOWN')
    
    # Formatting
    ax.set_title(f'Horizon {h}', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(transfer_springs)
    ax.set_ylabel('NSE Score')
    ax.set_xlabel('Spring ID')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add legend to the first subplot only to avoid clutter
    if i == 0:
        ax.legend(loc='lower right')

plt.suptitle('NSE Comparison across xLSTM Variants and Baselines', fontsize=18)
plt.tight_layout()
plt.savefig('evaluation/transfer/plots/xlstm_barplots_nofreeze.png')
plt.show()
