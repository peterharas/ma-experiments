import os
import pandas as pd
import matplotlib.pyplot as plt

# Custom colors
my_colors = {
    'LSTM': '#56B4E9',   # Sky Blue
    'xLSTM': '#0072B2',  # Deep Marine    
}

# 1. Load the data
df_xlstm = pd.read_csv('results/xLSTM_LARGE_results_20260612_065528.csv')
df_lstm = pd.read_csv('results/LSTM_LARGE_results_20260504_063436.csv')

df_xlstm['model'] = df_xlstm['model'].str.replace('_LARGE', '', regex=False)
df_lstm['model'] = df_lstm['model'].str.replace('_LARGE', '', regex=False)

# 2. Merge the dataframes on spring_id and horizon
df_merged = pd.merge(
    df_lstm, 
    df_xlstm, 
    on=['spring_id', 'horizon'], 
    suffixes=('_lstm', '_xlstm')
)

# 3. Sort the data by spring_id so the lines plot cleanly from left to right
df_merged = df_merged.sort_values(by=['spring_id'])

# 4. Convert spring_id to string so matplotlib treats it as categorical labels
df_merged['spring_id_str'] = df_merged['spring_id'].astype(str)

# Define horizons to iterate over
horizons = [1, 2, 3, 4]

# 5. Create subplots (stacked vertically, sharing the x-axis)
fig, axes = plt.subplots(nrows=len(horizons), ncols=1, figsize=(11.69, 8.27), sharex=True)

# Loop through each horizon and its corresponding axis
for ax, h in zip(axes, horizons):
    # Filter data for the current horizon
    df_h = df_merged[df_merged['horizon'] == h]
    
    # Plot LSTM and xLSTM lines using custom colors
    ax.plot(df_h['spring_id_str'], df_h['nse_lstm'], marker='o', 
            color=my_colors['LSTM'], label='LSTM', alpha=0.8)
    ax.plot(df_h['spring_id_str'], df_h['nse_xlstm'], marker='s', 
            color=my_colors['xLSTM'], label='xLSTM', alpha=0.8)
    
    # Formatting for each individual subplot
    ax.set_ylabel('NSE')
    ax.set_title(f'NSE Comparison: LSTM vs xLSTM generalised model (Horizon {h})')
    ax.legend(loc='lower left')
    ax.grid(True, linestyle='--', alpha=0.6)

# Formatting the shared x-axis (only applies to the bottom plot)
axes[-1].set_xlabel('Spring ID', fontsize=12)
plt.xticks(rotation=45, ha='right', rotation_mode='anchor')

plt.tight_layout()
plt.savefig(os.path.join('evaluation', 'xlstm_lstm_linecharts_large.png'), dpi=300, bbox_inches='tight')
plt.show()