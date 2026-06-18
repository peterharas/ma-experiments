import pandas as pd

# 1. Load the datasets (if not already loaded in your script)
df_xlstm = pd.read_csv('results/xLSTM_LARGE_results_20260612_065528.csv')
df_lstm = pd.read_csv('results/LSTM_LARGE_results_20260504_063436.csv')

# 2. Merge the datasets side-by-side on 'spring' and 'horizon'
# We only need the identifier, the horizon, and the NSE metric for this calculation
df_merged = pd.merge(
    df_xlstm[['spring_id', 'horizon', 'nse']], 
    df_lstm[['spring_id', 'horizon', 'nse']], 
    on=['spring_id', 'horizon'], 
    suffixes=('_xlstm', '_lstm')
)

# 3. Calculate who has the better NSE 
# Note: For Nash-Sutcliffe Efficiency (NSE), a higher value is better.
df_merged['xLSTM_better'] = df_merged['nse_xlstm'] > df_merged['nse_lstm']
df_merged['LSTM_better'] = df_merged['nse_lstm'] > df_merged['nse_xlstm']
df_merged['Tie'] = df_merged['nse_xlstm'] == df_merged['nse_lstm']

# 4. Group by the forecast horizon and count (sum the boolean True values)
win_counts = df_merged.groupby('horizon')[['xLSTM_better', 'LSTM_better', 'Tie']].sum().astype(int).reset_index()

# 5. Display the results
print("Number of springs where each model performs better (based on NSE):")
print(win_counts.to_string(index=False))

# Optional: Output to a markdown or LaTeX table for your report
# print(win_counts.to_latex(index=False))