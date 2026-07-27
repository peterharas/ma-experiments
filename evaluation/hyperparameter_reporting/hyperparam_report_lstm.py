import pandas as pd

df = pd.read_csv('results/LSTM_results_20260429_093001.csv')
df_filtered = df[df['horizon'] == 1]
cols_to_keep = ['spring_id', 'lstm_units', 'dropout', 'learning_rate', 'n_dense_layers']
df_table = df_filtered[cols_to_keep]

latex_table = df_table.to_latex(index=False)
print(latex_table)