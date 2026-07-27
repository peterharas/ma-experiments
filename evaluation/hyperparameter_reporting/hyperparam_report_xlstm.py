import pandas as pd

df = pd.read_csv('results/xLSTM_results_20260608_090620.csv')
df_filtered = df[df['horizon'] == 1]
cols_to_keep = ['spring_id', 'embedding_dim', 'dropout', 'learning_rate', 'architecture']
df_table = df_filtered[cols_to_keep]

latex_table = df_table.to_latex(index=False)
print(latex_table)