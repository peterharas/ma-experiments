import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the data
file_xlstm = 'results/xLSTM_LARGE_results_20260612_065528.csv'
file_lstm = 'results/LSTM_LARGE_results_20260504_063436.csv'

df_xlstm = pd.read_csv(file_xlstm)
df_lstm = pd.read_csv(file_lstm)

# 2. Merge the dataframes on spring_id and horizon
df_merged = pd.merge(
    df_lstm, 
    df_xlstm, 
    on=['spring_id', 'horizon'], 
    suffixes=('_lstm', '_xlstm')
)

# 3. Sort the data by spring_id so the lines plot cleanly from left to right
df_merged = df_merged.sort_values(by=['spring_id'])

# 4. Convert spring_id to string so matplotlib treats it as categorical labels, not numbers
df_merged['spring_id_str'] = df_merged['spring_id'].astype(str)

# 5. Loop through each horizon and create a separate plot
for h in [1, 2, 3, 4]:
    # Filter data for the current horizon
    df_h = df_merged[df_merged['horizon'] == h]
    
    # Create the plot
    plt.figure(figsize=(12, 5))
    
    # Plot LSTM and xLSTM lines
    plt.plot(df_h['spring_id_str'], df_h['nse_lstm'], marker='o', label='LSTM Large', alpha=0.8)
    plt.plot(df_h['spring_id_str'], df_h['nse_xlstm'], marker='s', label='xLSTM Large', alpha=0.8)
    
    # Formatting
    plt.xlabel('Spring ID')
    plt.ylabel('NSE')
    plt.title(f'NSE Comparison: LSTM Large vs xLSTM Large (Horizon {h})')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Rotate x-axis labels if there are many springs
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.show()