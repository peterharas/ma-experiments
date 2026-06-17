import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Custom colors
my_colors = {
    'LSTM': '#56B4E9',   # Sky Blue
    'xLSTM': '#0072B2',  # Deep Marine    
}

# 1. Load the data
file_xlstm = 'results/xLSTM_results_20260608_090620.csv'
file_lstm = 'results/LSTM_results_20260429_093001.csv'

df_xlstm = pd.read_csv(file_xlstm)
df_lstm = pd.read_csv(file_lstm)

# 2. Merge the dataframes on spring_id and horizon
df_merged = pd.merge(
    df_lstm, 
    df_xlstm, 
    on=['spring_id', 'horizon'], 
    suffixes=('_lstm', '_xlstm')
)

# 3. Sort the data by spring_id so the bars plot cleanly from left to right
df_merged = df_merged.sort_values(by=['spring_id'])

# 4. Convert spring_id to string so matplotlib treats it as categorical labels
df_merged['spring_id_str'] = df_merged['spring_id'].astype(str)

# Define horizons to iterate over
horizons = [1, 2, 3, 4]

# 5. Create subplots (stacked vertically, sharing the x-axis)
fig, axes = plt.subplots(nrows=len(horizons), ncols=1, figsize=(11.69, 8.27), sharex=True)

# Define the width of the bars
width = 0.35  

# Loop through each horizon and its corresponding axis
for ax, h in zip(axes, horizons):
    # Filter data for the current horizon
    df_h = df_merged[df_merged['horizon'] == h]
    
    # Create an array for numerical x-axis positions
    x = np.arange(len(df_h['spring_id_str']))
    
    # Plot LSTM and xLSTM grouped bars using custom colors and width
    ax.bar(x - width/2, df_h['nse_lstm'], width, 
           color=my_colors['LSTM'], label='LSTM', alpha=0.9)
    ax.bar(x + width/2, df_h['nse_xlstm'], width, 
           color=my_colors['xLSTM'], label='xLSTM', alpha=0.9)
    
    # Formatting for each individual subplot
    ax.set_ylabel('NSE')
    ax.set_title(f'NSE Comparison: LSTM vs xLSTM individual spring models (Horizon {h})')
    ax.legend(loc='lower right')
    
    # Grid lines are usually better on just the y-axis for bar charts
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)

# Formatting the shared x-axis (only applies to the bottom plot)
axes[-1].set_xlabel('Spring ID')
axes[-1].set_xticks(x)
axes[-1].set_xticklabels(df_h['spring_id_str'], rotation=45, ha='right', rotation_mode='anchor')

plt.tight_layout()

# Make sure the evaluation directory exists
os.makedirs('evaluation', exist_ok=True)
plt.savefig(os.path.join('evaluation', 'xlstm_lstm_barcharts_individual.png'), dpi=300, bbox_inches='tight')
plt.show()