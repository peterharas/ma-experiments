import os
import pandas as pd
import matplotlib.pyplot as plt

# Custom colors
my_colors = {
    'LSTM': '#56B4E9',   # Sky Blue
    'xLSTM': '#0072B2',  # Deep Marine
    'movingaverage': '#595959',
    'lastknown': '#BDBDBD'
}

# 1. Load the data
file_xlstm = 'results/xLSTM_results_20260608_090620.csv'
file_lstm = 'results/LSTM_results_20260429_093001.csv'

df_xlstm = pd.read_csv(file_xlstm)
df_lstm = pd.read_csv(file_lstm)

df_movingave = pd.read_csv('results/BASELINE_results_20260427_080304.csv')
df_lastknown = pd.read_csv('results/BASELINE_LASTKNOWN_results_20260427_080725.csv')
df_movingave['model'] = df_movingave['model'].str.replace('BASELINE', 'moving average', regex=False)
df_lastknown['model'] = df_lastknown['model'].str.replace('BASELINE_LASTKNOWN', 'last known', regex=False)

# 2. Merge the dataframes on spring_id and horizon
# 2.1 Set the index and append the specific suffixes to the metric columns of each dataframe
df1 = df_lstm.set_index(['spring_id', 'horizon']).add_suffix('_lstm')
df2 = df_xlstm.set_index(['spring_id', 'horizon']).add_suffix('_xlstm')
df3 = df_movingave.set_index(['spring_id', 'horizon']).add_suffix('_movingaverage')
df4 = df_lastknown.set_index(['spring_id', 'horizon']).add_suffix('_lastknown')
# 2.2 Join them all together and reset the index to turn spring_id and horizon back into columns
df_merged = df1.join([df2, df3, df4]).reset_index()

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
    
    # Plot the baselines
    ax.plot(df_h['spring_id_str'], df_h['nse_movingaverage'], marker='s', 
        color=my_colors['movingaverage'], label='Moving Average', alpha=0.8)
    ax.plot(df_h['spring_id_str'], df_h['nse_lastknown'], marker='s', 
        color=my_colors['lastknown'], label='Last Known', alpha=0.8)
    
    # Formatting for each individual subplot
    ax.set_ylabel('NSE')
    ax.set_title(f'NSE Comparison: LSTM vs xLSTM individual spring models (Horizon {h})')
    ax.legend(loc='lower right')
    ax.grid(True, linestyle='--', alpha=0.6)

# Formatting the shared x-axis (only applies to the bottom plot)
axes[-1].set_xlabel('Spring ID', fontsize=12)
plt.xticks(rotation=45, ha='right', rotation_mode='anchor')

plt.tight_layout()
plt.savefig(os.path.join('evaluation', 'xlstm_lstm_linecharts_individual.png'), dpi=300, bbox_inches='tight')
plt.show()