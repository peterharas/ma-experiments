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
df_tft_individual = pd.read_csv('results/TFT_WEATHER_results_20260710_122632.csv')
df_tft_large = pd.read_csv('results/TFT_LARGE_WEATHER_results_20260716_184756.csv')
df_tft_transfer = pd.read_csv('results/TFT_TRANSFER_results_20260717_205404.csv')

transfer_springs = sorted(df_tft_transfer['spring_id'].unique())

# Filter all dataframes to ensure we only plot these 7 springs
df_base = df_base[df_base['spring_id'].isin(transfer_springs)]
df_base_lk = df_base_lk[df_base_lk['spring_id'].isin(transfer_springs)]
df_tft_individual = df_tft_individual[df_tft_individual['spring_id'].isin(transfer_springs)]
df_tft_large = df_tft_large[df_tft_large['spring_id'].isin(transfer_springs)]

# Setup a 2x2 grid for the 4 horizons
fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharey=True)
axes = axes.flatten()

horizons = [1, 2, 3, 4]

bar_color = '#BB0000'   # Same blue for all bars
edge_color = 'black'

x = np.arange(len(transfer_springs))
width = 0.25

for i, h in enumerate(horizons):
    ax = axes[i]
    
    # Extract data for the current horizon and align by spring_id
    h_tft_i = df_tft_individual[df_tft_individual['horizon'] == h].set_index('spring_id')['nse'].reindex(transfer_springs).values
    h_tft_l = df_tft_large[df_tft_large['horizon'] == h].set_index('spring_id')['nse'].reindex(transfer_springs).values
    h_tft_t = df_tft_transfer[df_tft_transfer['horizon'] == h].set_index('spring_id')['nse'].reindex(transfer_springs).values
    
    h_base = df_base[df_base['horizon'] == h].set_index('spring_id')['nse'].reindex(transfer_springs).values
    h_base_lk = df_base_lk[df_base_lk['horizon'] == h].set_index('spring_id')['nse'].reindex(transfer_springs).values
    
    # Plot grouped bars
    ax.bar(
        x - width, h_tft_i, width,
        label='TFT Individual',
        color=bar_color,
        edgecolor=edge_color,
        hatch='',
        linewidth=1
    )

    ax.bar(
        x, h_tft_l, width,
        label='TFT Multi-spring',
        color=bar_color,
        edgecolor=edge_color,
        hatch='////',
        linewidth=1
    )

    ax.bar(
        x + width, h_tft_t, width,
        label='TFT Fine-tuned',
        color=bar_color,
        edgecolor=edge_color,
        hatch='xxxx',
        linewidth=1
    )
    
    # Plot lines for baselines
    ax.plot(x, h_base, color='dimgray', linestyle='-', marker='o', linewidth=2, label='Moving Average Baseline')
    ax.plot(x, h_base_lk, color='silver', linestyle='--', marker='x', linewidth=2, label='Last Known Baseline')
    
    # Formatting
    ax.set_title(f'Horizon {h}', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(transfer_springs)
    ax.set_ylabel('NSE Score')
    ax.set_xlabel('Spring ID')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    ax.legend(loc='lower right')

plt.tight_layout()
plt.savefig('evaluation/transfer/plots/tft_barplots.png')
plt.show()
