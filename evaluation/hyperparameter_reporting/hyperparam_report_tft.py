import pandas as pd

df = pd.read_csv('results/TFT_results_20260630_092241.csv')
df_filtered = df[df['horizon'] == 1]
cols_to_keep = ['spring_id', 'hidden_size', 'learning_rate', 'lstm_layers']
df_table = df_filtered[cols_to_keep]

latex_table = df_table.to_latex(index=False)
print(latex_table)