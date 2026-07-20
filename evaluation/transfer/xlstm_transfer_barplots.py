import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.size': 14,          # Base font size
    'axes.titlesize': 16,     # Subplot title size
    'axes.labelsize': 14,     # X/Y label size
    'xtick.labelsize': 14,    # X tick label size
    'ytick.labelsize': 14,    # Y tick label size
    'legend.fontsize': 14     # Legend font size
})

# Load data
df_base = pd.read_csv('results/BASELINE_results_20260427_080304.csv')
df_base_lk = pd.read_csv('results/BASELINE_LASTKNOWN_results_20260427_080725.csv')
df_xlstm = pd.read_csv('results/xLSTM_results_20260608_090620.csv')
df_xlstm_large = pd.read_csv('results/xLSTM_LARGE_results_20260612_065528.csv')
df_xlstm_transfer = pd.read_csv('results/xLSTM_TRANSFER_results_20260715_153842.csv')

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

bar_color = '#0072B2'   # Same blue for all bars
edge_color = 'black'

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
    ax.bar(
        x - width, h_xlstm, width,
        label='xLSTM Individual',
        color=bar_color,
        edgecolor=edge_color,
        hatch='',
        linewidth=1
    )

    ax.bar(
        x, h_xlstm_l, width,
        label='xLSTM Multi-spring',
        color=bar_color,
        edgecolor=edge_color,
        hatch='////',
        linewidth=1
    )

    ax.bar(
        x + width, h_xlstm_t, width,
        label='xLSTM Fine-tuned',
        color=bar_color,
        edgecolor=edge_color,
        hatch='xxxx',
        linewidth=1
    )
    
    # Plot lines for baselines
    ax.plot(x, h_base, color='dimgray', linestyle='-', marker='o', linewidth=2, label='Moving Average Baseline')
    ax.plot(x, h_base_lk, color='silver', linestyle='--', marker='x', linewidth=2, label='Last Known Baseline')
    
    # Formatting
    ax.set_title(f'Horizon {h} Day', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(transfer_springs)
    ax.set_ylabel('NSE Score')
    ax.set_xlabel('Spring ID')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add legend to the first subplot only to avoid clutter
    ax.legend(loc='lower right')

# plt.suptitle('NSE Comparison across xLSTM Variants and Baselines', fontsize=18)
plt.tight_layout()
plt.savefig('evaluation/transfer/plots/xlstm_barplots_mlstm.png')
plt.show()
