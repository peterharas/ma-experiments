import os
import pandas as pd
import matplotlib.pyplot as plt

my_colors = {
    'TFT_WEATHER': '#BB0000',
    'TFT': '#D55E00'
}

plt.rcParams.update({
    'font.size': 12,          # Base font size
    'axes.titlesize': 14,     # Subplot title size (default is 12)
    'axes.labelsize': 12,     # X/Y label size (default is 10)
    'xtick.labelsize': 12,    # X tick label size (default is 10)
    'ytick.labelsize': 12,    # Y tick label size (default is 10)
    'legend.fontsize': 12     # Legend font size (default is 10)
})

# 1. Load the data
df_tft = pd.read_csv('results/TFT_results_20260630_092241.csv')
df_weather = pd.read_csv('results/TFT_WEATHER_results_20260710_122632.csv')


# 2. Merge the dataframes on spring_id and horizon
df_merged = pd.merge(
    df_weather, 
    df_tft, 
    on=['spring_id', 'horizon'], 
    suffixes=('_tft_weather', '_tft')
)

# 3. Sort the data by spring_id so the lines plot cleanly from left to right
df_merged = df_merged.sort_values(by=['spring_id'])

# 4. Convert spring_id to string so matplotlib treats it as categorical labels
df_merged['spring_id_str'] = df_merged['spring_id'].astype(str)

# Define horizons to iterate over
horizons = [1, 2, 3, 4]

# 6. Paired difference plot: ΔNSE = TFT - TFT_WEATHER

fig, axes = plt.subplots(
    nrows=len(horizons), 
    ncols=1, 
    figsize=(11.69, 8.27), 
    sharex=True,
    sharey=True
)

for ax, h in zip(axes, horizons):
    # Filter horizon
    df_h = df_merged[df_merged['horizon'] == h].copy()
    
    # Compute difference
    df_h['delta_nse'] = df_h['nse_tft_weather'] - df_h['nse_tft']
    
    # Sort by performance difference
    df_h = df_h.sort_values(by='delta_nse')
    
    # Use categorical x-axis in sorted order
    x = df_h['spring_id_str']
    y = df_h['delta_nse']
   

    # Lollipop plot
    ax.vlines(x, 0, y, color=my_colors['TFT_WEATHER'], alpha=0.8)
    ax.scatter(x, y, color=my_colors['TFT_WEATHER'])
    
    # Zero reference line
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    
    # Formatting
    ax.set_ylabel('Δ NSE')
    ax.set_title(f'Paired Difference (TFT_WEATHER − TFT) | Individual Spring Models | Horizon {h}')
    ax.grid(True, linestyle='--', alpha=0.6)

# Shared x-axis formatting
axes[-1].set_xlabel('Spring ID', fontsize=12)
plt.xticks(rotation=45, ha='right', rotation_mode='anchor')

plt.tight_layout()
plt.savefig(os.path.join('evaluation', 'tft_weather', 'plots', 'tft_weather_lollipop.png'), dpi=300, bbox_inches='tight')
plt.show()